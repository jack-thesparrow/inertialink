// tui_main.cpp — Inertialink Terminal UI
//
// lazygit-style launcher for the Inertialink smart pen project.
// Fully responsive (re-lays out on every resize event via FTXUI).
//
// Panels  (Tab cycles through them):
//   0  Connection   — Radiobox: USB / WiFi / Simulation
//   1  Actions      — 6 tools; ↑↓ navigate, Space/Enter toggle run·stop
//   2  Mock input   — word / 'all' text field (only when Mock ESP32 on cursor)
//   3  Test panel   — 12-word grid; ↑↓←→ navigate, Space mark, Enter stream, A all
//   4  Output/Log   — per-process output sections, auto-split by active processes
//
// Key map (global):
//   1-4          switch panel (Connection / Actions / Test / Output)
//   Tab          cycle panels
//   ↑↓ / ←→     navigate within focused panel
//   Space        run/stop action  |  mark/unmark word for batch test
//   Enter        run/stop action  |  stream highlighted word immediately
//   A            stream marked words  (all 12 if none marked)
//   K            kill every running process
//   C            clear the log
//   PgUp/PgDn    scroll output
//   ?            toggle keyboard-reference overlay
//   Q            quit + SIGTERM all processes

#include "pen/io.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/terminal.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace ftxui;

// ── Shared log ────────────────────────────────────────────────────────────────
struct LogLine { std::string ts, src, msg; };

static std::mutex          gLogMtx;
static std::deque<LogLine> gLog;
static constexpr int       LOG_CAP = 500;

static std::string nowTs() {
  auto t = std::time(nullptr);
  struct tm tm{};
  localtime_r(&t, &tm);
  char buf[9];
  std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
  return buf;
}
static void logPush(const std::string &src, const std::string &msg) {
  std::lock_guard lk(gLogMtx);
  gLog.push_back({nowTs(), src, msg});
  if (static_cast<int>(gLog.size()) > LOG_CAP) gLog.pop_front();
}

// ── Process management ────────────────────────────────────────────────────────
struct Proc { std::string name; pid_t pid = -1; int rfd = -1; bool alive = false; };

static Proc procLaunch(const std::string &name, std::vector<std::string> args) {
  int fds[2];
  if (::pipe(fds) < 0) { logPush("tui", "pipe() failed"); return {}; }
  pid_t pid = ::fork();
  if (pid < 0) { ::close(fds[0]); ::close(fds[1]); logPush("tui","fork() failed"); return {}; }
  if (pid == 0) {
    ::dup2(fds[1], STDOUT_FILENO); ::dup2(fds[1], STDERR_FILENO);
    ::close(fds[0]); ::close(fds[1]);
    // Force Python to flush every print() immediately when piped.
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    ::execvp(argv[0], argv.data());
    ::_exit(127);
  }
  ::close(fds[1]);
  ::fcntl(fds[0], F_SETFL, O_NONBLOCK);
  logPush("tui", "Started " + name + " (pid " + std::to_string(pid) + ")");
  return {name, pid, fds[0], true};
}
static void procStop(Proc &p) {
  if (!p.alive) return;
  if (p.pid > 0)  { ::kill(p.pid, SIGTERM); ::waitpid(p.pid, nullptr, WNOHANG); }
  if (p.rfd >= 0) ::close(p.rfd);
  logPush("tui", "Stopped " + p.name);
  p = {};
}
static bool procDrainAll(std::vector<Proc> &procs) {
  bool any = false; char buf[1024];
  for (auto &p : procs) {
    if (p.rfd < 0) continue;
    std::string chunk; ssize_t n;
    while ((n = ::read(p.rfd, buf, sizeof(buf)-1)) > 0) { buf[n]='\0'; chunk+=buf; any=true; }
    std::istringstream ss(chunk);
    for (std::string line; std::getline(ss, line);)
      if (!line.empty()) logPush(p.name, line);
    if (p.pid > 0) {
      int st;
      if (::waitpid(p.pid, &st, WNOHANG) > 0) {
        logPush("tui", p.name + " exited (code " + std::to_string(WEXITSTATUS(st)) + ")");
        ::close(p.rfd); p.rfd=-1; p.pid=-1; p.alive=false;
      }
    }
  }
  return any;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
  auto screen = ScreenInteractive::Fullscreen();

  // ── Connection ─────────────────────────────────────────────────────────────
  int conn = 0;
  const std::vector<std::string> kConnLabels = {
      std::string("USB        (") + pen::Defaults::usbPort + ")",
      "WiFi       (port " + std::to_string(pen::Defaults::wifiPort)
                          + "/" + std::to_string(pen::Defaults::wifiVizPort) + ")",
      "Simulation (/tmp/vtty_laptop)",
  };
  const std::vector<std::string> kConnArgs = {"usb", "wifi", "sim"};

  // ── Actions ────────────────────────────────────────────────────────────────
  struct ActionDef {
    std::string name, hint;
    std::vector<std::string> cmd;
    bool useConnArg;
  };
  const std::vector<ActionDef> kActions = {
    { "Visualizer", "3D cube + stroke canvas",
      {"./bin/visualizer"}, true },
    { "Decoder",    "Real-time AI recognition",
      {"./bin/decoder"}, true },
    { "Collector",  "Record IMU data to CSV",
      {"./bin/data_collector"}, true },
    { "Mock ESP32", "",
      {".venv/bin/python3","scripts/mock_esp32.py"}, false },
    { "Train",      "BiLSTM CTC model training",
      {".venv/bin/python3","scripts/train_bilstm.py"}, false },
    { "Evaluate",   "Batch accuracy on all samples",
      {".venv/bin/python3","scripts/eval_model.py"}, false },
  };
  static constexpr int MOCK_IDX = 3;

  // ── Trained words (matches mock_esp32.py TRAINED_WORDS) ───────────────────
  const std::vector<std::string> kWords = {
      "hello","world","pen","123","write","note",
      "data","code","test","abc","xyz","open"
  };
  static constexpr int WORD_COLS = 4;
  int testWord = 0;

  // ── State ──────────────────────────────────────────────────────────────────
  int  act       = 0;
  int  focus     = 1;   // 0=conn 1=actions 2=mock-input 3=test 4=log
  int  ftFocus   = 0;
  int  logScroll = 0;
  std::vector<Proc> procs(kActions.size());
  std::string   mockWord    = "hello";
  std::string   nowTesting;
  std::set<int> selWords;   // test-panel words marked with Space
  bool          showHelp    = false;

  auto syncFt = [&] {
    if      (focus == 0) ftFocus = 0;  // connRadio receives keyboard
    else if (focus == 2) ftFocus = 1;  // mockInput receives keyboard
    else                 ftFocus = 2;  // dummyComp — won't steal arrow keys
  };
  syncFt();

  // ── Launch / stop ──────────────────────────────────────────────────────────
  auto toggleAction = [&](int i) {
    if (procs[i].alive) { procStop(procs[i]); return; }
    auto cmd = kActions[i].cmd;
    if (kActions[i].useConnArg) cmd.push_back(kConnArgs[conn]);
    if (i == MOCK_IDX) cmd.push_back(mockWord.empty() ? "hello" : mockWord);
    procs[i] = procLaunch(kActions[i].name, cmd);
  };

  // Stream a single word via mock_esp32
  auto streamWord = [&](const std::string &word) {
    if (procs[MOCK_IDX].alive) procStop(procs[MOCK_IDX]);
    auto cmd = kActions[MOCK_IDX].cmd;
    cmd.push_back(word);
    procs[MOCK_IDX] = procLaunch("Mock", cmd);
    nowTesting = (word == "all") ? "all 12 words" : word;
  };

  // Stream a set of words (each as a separate argument; mock handles the list)
  auto streamWords = [&](const std::vector<std::string> &words) {
    if (procs[MOCK_IDX].alive) procStop(procs[MOCK_IDX]);
    auto cmd = kActions[MOCK_IDX].cmd;
    for (auto &w : words) cmd.push_back(w);
    nowTesting = words.size() == 1
        ? words[0]
        : std::to_string(words.size()) + " selected words";
    procs[MOCK_IDX] = procLaunch("Mock", cmd);
  };

  // ── FTXUI components ───────────────────────────────────────────────────────
  auto connRadio = Radiobox(&kConnLabels, &conn);
  auto mockInput = Input(&mockWord, "word / 'all'");
  auto dummyComp = Renderer([] { return text(""); });
  auto layout    = Container::Tab({connRadio, mockInput, dummyComp, dummyComp}, &ftFocus);

  // ── Renderer ───────────────────────────────────────────────────────────────
  auto ui = Renderer(layout, [&]() -> Element {
    // ── helpers ──────────────────────────────────────────────────────────────
    auto kd = [](const std::string &k, const std::string &desc) {
      return hbox({
          text(" " + k + " ") | bold | bgcolor(Color::Blue) | color(Color::White),
          text(" " + desc + " "),
      });
    };
    // Panel title: yellow when focused, white otherwise.
    auto pTitle = [&](int panelFocus, const std::string &label) -> Element {
      return text(label) | bold
           | ((focus == panelFocus) ? color(Color::Yellow) : color(Color::White));
    };

    // ── Left column ───────────────────────────────────────────────────────────
    auto connBox = window(pTitle(0, " [1] Connection "), connRadio->Render());

    Elements dotRows;
    for (int i = 0; i < static_cast<int>(kActions.size()); ++i) {
      auto dot = procs[i].alive
          ? (text("● ") | color(Color::Green))
          : (text("○ ") | color(Color::GrayDark));
      dotRows.push_back(hbox({dot, text(kActions[i].name)}));
    }
    auto statusBox = window(text(" Status ") | bold, vbox(std::move(dotRows)));
    auto leftCol   = vbox({connBox, statusBox});

    // ── Actions panel ─────────────────────────────────────────────────────────
    Elements actRows;
    for (int i = 0; i < static_cast<int>(kActions.size()); ++i) {
      bool sel    = (i == act) && (focus == 1);
      auto numTag = text(" " + std::to_string(i+1) + " ") | color(Color::Yellow) | bold;
      auto tag    = procs[i].alive
          ? (text(" STOP ") | color(Color::Red)   | bold)
          : (text(" RUN  ") | color(Color::Green) | bold);
      std::string hint = (i == MOCK_IDX)
          ? ("word: " + (mockWord.empty() ? "hello" : mockWord))
          : kActions[i].hint;
      auto row = hbox({numTag, tag, text("  "),
                       text(kActions[i].name) | size(WIDTH, EQUAL, 11),
                       text("  "), text(hint) | dim});
      actRows.push_back(sel ? (row | inverted) : row);
    }
    auto mockRow = hbox({
        text(" Word: ") | (focus == 2 ? bold : dim),
        mockInput->Render() | size(WIDTH, EQUAL, 24),
        text("  [Tab] to type") | dim,
    });
    auto actionsBox = window(
        pTitle(1, " [2] Actions   ↑↓ navigate   Space/Enter run·stop "),
        vbox({vbox(std::move(actRows)), separator(), mockRow}));

    // ── Test panel ────────────────────────────────────────────────────────────
    Elements wordRows;
    int nWords = static_cast<int>(kWords.size());
    for (int row = 0; row * WORD_COLS < nWords; ++row) {
      Elements cols;
      for (int c = 0; c < WORD_COLS; ++c) {
        int  idx    = row * WORD_COLS + c;
        if (idx >= nWords) { cols.push_back(text("") | flex); continue; }
        bool cursor = (idx == testWord) && (focus == 3);
        bool marked = selWords.count(idx) > 0;
        auto mark   = marked ? (text("✓") | color(Color::Green) | bold) : text(" ");
        auto cell   = hbox({
            mark,
            text(std::to_string(idx+1)) | color(Color::Yellow) | size(WIDTH, EQUAL, 2),
            text(" "),
            text(kWords[idx]) | size(WIDTH, EQUAL, 6),
        }) | size(WIDTH, EQUAL, 12);
        cols.push_back(cursor ? (cell | inverted) : cell);
      }
      wordRows.push_back(hbox(std::move(cols)));
    }
    auto testBox = window(
        pTitle(3, " [3] Test   ↑↓←→ navigate   Space mark   Enter stream   A all "),
        vbox(std::move(wordRows)));

    // ── Per-process output sections ───────────────────────────────────────────
    // Copy log data under the mutex, grouped by source in first-seen order.
    struct SrcBuf { std::string name; std::vector<LogLine> lines; };
    std::vector<SrcBuf> srcBufs;
    {
      std::unordered_map<std::string, int> idx;
      std::lock_guard lk(gLogMtx);
      for (auto &l : gLog) {
        auto it = idx.find(l.src);
        if (it == idx.end()) {
          idx[l.src] = static_cast<int>(srcBufs.size());
          srcBufs.push_back({l.src, {}});
        }
        srcBufs[idx[l.src]].lines.push_back(l);
      }
      logScroll = std::min(logScroll, std::max(0, static_cast<int>(gLog.size())-1));
    }

    // Separate "tui" system events (small header) from process outputs (flex sections).
    SrcBuf *tuiBuf = nullptr;
    std::vector<SrcBuf *> procBufs;
    for (auto &sb : srcBufs) {
      if (sb.name == "tui") tuiBuf = &sb;
      else                  procBufs.push_back(&sb);
    }

    // Estimate lines available for the output zone based on terminal height.
    auto termSz    = Terminal::Size();
    int  outHeight = std::max(6, termSz.dimy - 22); // rows below actions+test+bars
    int  numProcs  = std::max(1, static_cast<int>(procBufs.size()));
    int  perProc   = std::max(2, outHeight / numProcs - 3); // -3 for border + title

    Elements sections;

    // System-events strip (last 2 "tui" lines, fixed, no flex)
    if (tuiBuf && !tuiBuf->lines.empty()) {
      auto &tl  = tuiBuf->lines;
      int   tot = static_cast<int>(tl.size());
      Elements sysRows;
      for (int i = std::max(0, tot-2); i < tot; ++i)
        sysRows.push_back(hbox({
            text(tl[i].ts) | color(Color::GrayDark),
            text("  "),
            text(tl[i].msg) | dim | flex,
        }));
      sections.push_back(hbox({
          text(" sys ") | dim,
          vbox(std::move(sysRows)) | flex,
      }));
    }

    // One flex section per process that produced output.
    if (procBufs.empty()) {
      sections.push_back(
          window(pTitle(4, " [4] Output "),
                 text("Waiting for output...") | dim | center) | flex);
    } else {
      bool focusOutput = (focus == 4);
      for (auto *sb : procBufs) {
        int   tot   = static_cast<int>(sb->lines.size());
        int   end   = std::max(0, tot - logScroll);
        int   start = std::max(0, end - perProc);
        Elements rows;
        for (int i = start; i < end; ++i)
          rows.push_back(hbox({
              text(sb->lines[i].ts) | color(Color::GrayDark),
              text(" "),
              text(sb->lines[i].msg) | flex,
          }));
        if (rows.empty()) rows.push_back(text("(no output yet)") | dim);

        Color titleColor = focusOutput ? Color::Yellow : Color::Cyan;
        sections.push_back(
            window(text(" " + sb->name + " ") | bold | color(titleColor),
                   vbox(std::move(rows)) | flex) | flex);
      }
    }

    auto outputArea = vbox(std::move(sections)) | flex;

    // "Now testing" indicator between test panel and output
    Element testingRow = nowTesting.empty()
        ? (text("") | size(HEIGHT, EQUAL, 0))
        : hbox({text(" ↗ Now testing: ") | bold | color(Color::Cyan),
                text(nowTesting) | bold | color(Color::Yellow)});

    auto rightCol = vbox({actionsBox, testBox, testingRow, outputArea}) | flex;

    // ── Bottom bar ────────────────────────────────────────────────────────────
    auto bar = hbox({
        kd("1-4",     "panel"),
        kd("Tab",     "cycle"),
        kd("↑↓",      "nav"),
        kd("Space",   "select"),
        kd("Enter",   "run"),
        kd("A",       "all"),
        kd("K",       "kill"),
        kd("C",       "clear"),
        kd("PgUp/Dn", "scroll"),
        kd("?",       "help"),
        kd("Q",       "quit"),
    });

    auto mainLayout = vbox({
        text(" ★  Inertialink Smart Pen") | bold | center,
        separator(),
        hbox({leftCol, separator(), rightCol}) | flex,
        separator(),
        bar,
    });

    // ── Help overlay (dbox on top of main layout) ─────────────────────────────
    if (!showHelp) return mainLayout;

    auto helpBody = vbox({
        text(" Keyboard Reference ") | bold | color(Color::Yellow) | center,
        separator(),
        hbox({text("  1-4       ") | bold | color(Color::Yellow),
              text("Switch panel  Connection / Actions / Test / Output")}),
        hbox({text("  Tab       ") | bold | color(Color::Yellow),
              text("Cycle panels forward")}),
        hbox({text("  ↑↓ / ←→  ") | bold | color(Color::Yellow),
              text("Navigate within focused panel")}),
        hbox({text("  Space     ") | bold | color(Color::Yellow),
              text("Toggle run/stop (Actions)  |  mark/unmark word (Test)")}),
        hbox({text("  Enter     ") | bold | color(Color::Yellow),
              text("Toggle run/stop (Actions)  |  stream word (Test)")}),
        hbox({text("  A         ") | bold | color(Color::Yellow),
              text("Stream marked words, or all 12 if none marked")}),
        hbox({text("  K         ") | bold | color(Color::Yellow),
              text("Kill all running processes")}),
        hbox({text("  C         ") | bold | color(Color::Yellow),
              text("Clear output log")}),
        hbox({text("  PgUp/PgDn ") | bold | color(Color::Yellow),
              text("Scroll output")}),
        hbox({text("  Q         ") | bold | color(Color::Yellow),
              text("Quit  (SIGTERM all processes)")}),
        hbox({text("  ?         ") | bold | color(Color::Yellow),
              text("Toggle this help overlay")}),
        separator(),
        text("  Press any key to dismiss  ") | dim | center,
    });
    auto helpBox = window(text(""), helpBody) | clear_under | center;
    return dbox({mainLayout, helpBox});
  });

  // ── Event handler ──────────────────────────────────────────────────────────
  auto withEvents = CatchEvent(ui, [&](Event ev) -> bool {
    // Any key dismisses the help overlay.
    if (showHelp) { showHelp = false; return true; }

    // '?' opens help (guard: not while typing)
    if (focus != 2 && ev == Event::Character('?')) {
      showHelp = true; return true;
    }

    // ── Quit ─────────────────────────────────────────────────────────────────
    if (ev == Event::Character('q') || ev == Event::Character('Q')) {
      screen.ExitLoopClosure()(); return true;
    }

    // ── Panel switch 1-4 ─────────────────────────────────────────────────────
    if (focus != 2) {
      if (ev == Event::Character('1')) { focus = 0; syncFt(); return true; }
      if (ev == Event::Character('2')) { focus = 1; syncFt(); return true; }
      if (ev == Event::Character('3')) { focus = 3; syncFt(); return true; }
      if (ev == Event::Character('4')) { focus = 4; syncFt(); return true; }
    }

    // ── Kill all ─────────────────────────────────────────────────────────────
    if (ev == Event::Character('k') || ev == Event::Character('K')) {
      for (auto &p : procs) if (p.alive) procStop(p);
      return true;
    }

    // ── Clear log ────────────────────────────────────────────────────────────
    if (focus != 2 && (ev == Event::Character('c') || ev == Event::Character('C'))) {
      std::lock_guard lk(gLogMtx); gLog.clear();
      selWords.clear(); nowTesting.clear();
      return true;
    }

    // ── Scroll ───────────────────────────────────────────────────────────────
    if (ev == Event::PageUp)   { logScroll += 5;                      return true; }
    if (ev == Event::PageDown) { logScroll = std::max(0,logScroll-5); return true; }

    // ── Tab cycle ────────────────────────────────────────────────────────────
    if (ev == Event::Tab) {
      if      (focus == 0)                     focus = 1;
      else if (focus == 1 && act == MOCK_IDX)  focus = 2;
      else if (focus == 1 || focus == 2)        focus = 3;
      else if (focus == 3)                      focus = 4;
      else                                      focus = 0;
      syncFt(); return true;
    }

    // ── Actions panel ────────────────────────────────────────────────────────
    if (focus == 1) {
      if (ev == Event::ArrowUp)   { act = std::max(0, act-1); return true; }
      if (ev == Event::ArrowDown) {
        act = std::min(static_cast<int>(kActions.size())-1, act+1); return true;
      }
      // Space and Enter both toggle run/stop
      if (ev == Event::Return || ev == Event::Character(' ')) {
        toggleAction(act); return true;
      }
    }

    // ── Test panel ───────────────────────────────────────────────────────────
    if (focus == 3) {
      int row    = testWord / WORD_COLS;
      int col    = testWord % WORD_COLS;
      int maxRow = (static_cast<int>(kWords.size())-1) / WORD_COLS;
      int nWords = static_cast<int>(kWords.size());

      // Always consume arrow keys so they can't leak to other panels.
      if (ev == Event::ArrowUp)    { if (row > 0) testWord -= WORD_COLS;                             return true; }
      if (ev == Event::ArrowDown)  { if (row < maxRow) testWord = std::min(nWords-1, testWord+WORD_COLS); return true; }
      if (ev == Event::ArrowLeft)  { if (col > 0) testWord -= 1;                                     return true; }
      if (ev == Event::ArrowRight) { if (col < WORD_COLS-1 && testWord+1 < nWords) testWord += 1;   return true; }

      // Space: mark / unmark the current word
      if (ev == Event::Character(' ')) {
        if (selWords.count(testWord)) selWords.erase(testWord);
        else selWords.insert(testWord);
        return true;
      }
      // Enter: stream the highlighted word immediately
      if (ev == Event::Return) {
        streamWord(kWords[testWord]); return true;
      }
      // A: stream marked words, or all if nothing marked
      if (ev == Event::Character('a') || ev == Event::Character('A')) {
        if (!selWords.empty()) {
          std::vector<std::string> wl;
          for (int w : selWords) wl.push_back(kWords[w]);
          streamWords(wl);
        } else {
          streamWord("all");
        }
        return true;
      }
    }

    // ── Output/Log panel — consume arrows so they don't reach connRadio ─────────
    if (focus == 4) {
      if (ev == Event::ArrowUp || ev == Event::ArrowDown ||
          ev == Event::ArrowLeft || ev == Event::ArrowRight)
        return true;
    }

    return false;
  });

  // ── Background pipe-drain thread ───────────────────────────────────────────
  std::atomic<bool> bgStop{false};
  std::thread bgThread([&] {
    while (!bgStop) {
      if (procDrainAll(procs)) screen.PostEvent(Event::Custom);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  logPush("tui", "Ready.  1-4 panel | Tab cycle | Space select | Enter run | ? help | Q quit");
  screen.Loop(withEvents);

  bgStop = true;
  bgThread.join();
  for (auto &p : procs) if (p.alive) procStop(p);
  return 0;
}
