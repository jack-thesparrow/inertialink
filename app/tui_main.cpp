// tui_main.cpp — Inertialink Terminal UI
//
// lazygit-style launcher for the Inertialink smart pen project.
// Fully responsive (re-lays out on every resize event via FTXUI).
//
// Panels  (Tab cycles through them):
//   0  Connection   — Radiobox: WiFi / USB / Bluetooth / Simulation
//   1  Actions      — 6 tools; number keys 1-6 jump directly to any action
//   2  Mock input   — word / 'all' text field (only reachable when Mock ESP32 selected)
//   3  Test panel   — 12-word grid; ↑↓←→ navigate, Enter stream, A stream all
//
// Key map (global):
//   1-6          jump to action (actions panel)
//   Tab          cycle panels
//   ↑↓ / ←→     navigate within focused panel
//   Enter        launch / stop  (actions)  or  stream word  (test panel)
//   A            stream all words  (test panel only)
//   K            kill every running process
//   C            clear the log
//   PgUp/PgDn    scroll log
//   Q            quit + SIGTERM all processes

#include "pen/io.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
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
      "WiFi       (port " + std::to_string(pen::Defaults::wifiPort)
                          + "/" + std::to_string(pen::Defaults::wifiVizPort) + ")",
      std::string("USB        (") + pen::Defaults::usbPort + ")",
      std::string("Bluetooth  (") + pen::Defaults::bluetoothPort + ")",
      "Simulation (/tmp/vtty_laptop)",
  };
  const std::vector<std::string> kConnArgs = {"wifi","usb","bt","sim"};

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
    { "Mock ESP32", "",   // hint shows word dynamically
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
  static constexpr int WORD_COLS = 4;   // grid columns
  int testWord = 0;                      // 0-11

  // ── State ──────────────────────────────────────────────────────────────────
  int act        = 0;
  int focus      = 1;   // 0=conn, 1=actions, 2=mock-input, 3=test-panel
  int ftFocus    = 0;   // index into Container::Tab children
  int logScroll  = 0;   // lines scrolled up from bottom
  std::vector<Proc> procs(kActions.size());
  std::string mockWord = "hello";

  // Map our 4-panel focus to the two real FTXUI components
  auto syncFt = [&] {
    ftFocus = (focus == 0) ? 0 : (focus == 2) ? 1 : 0;
    // When focus==1 or 3, neither FTXUI component owns the keyboard;
    // CatchEvent handles it directly.
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

  // Run mock with a specific word (stops current mock first)
  auto streamWord = [&](const std::string &word) {
    if (procs[MOCK_IDX].alive) procStop(procs[MOCK_IDX]);
    auto cmd = kActions[MOCK_IDX].cmd;
    cmd.push_back(word);
    procs[MOCK_IDX] = procLaunch("Mock:" + word, cmd);
  };

  // ── FTXUI components ───────────────────────────────────────────────────────
  auto connRadio = Radiobox(&kConnLabels, &conn);
  auto mockInput = Input(&mockWord, "word / 'all'");

  // Dummy focusable component to hold the actions/test focus slots
  auto dummyComp = Renderer([] { return text(""); });

  auto layout = Container::Tab({connRadio, mockInput, dummyComp, dummyComp}, &ftFocus);

  // ── Renderer ───────────────────────────────────────────────────────────────
  auto ui = Renderer(layout, [&]() -> Element {
    // ── helpers ──────────────────────────────────────────────────────────────
    auto kd = [](const std::string &k, const std::string &desc) {
      return hbox({
          text(" " + k + " ") | bold | bgcolor(Color::Blue) | color(Color::White),
          text(" " + desc + "  "),
      });
    };

    // ── Left: connection + status ─────────────────────────────────────────
    auto connBox = window(
        text(" Connection ") | bold,
        connRadio->Render());

    Elements dotRows;
    for (int i = 0; i < static_cast<int>(kActions.size()); ++i) {
      auto dot = procs[i].alive
          ? (text("● ") | color(Color::Green))
          : (text("○ ") | color(Color::GrayDark));
      dotRows.push_back(hbox({dot, text(kActions[i].name)}));
    }
    auto statusBox = window(text(" Status ") | bold, vbox(std::move(dotRows)));

    auto leftCol = vbox({connBox, statusBox});

    // ── Right top: action list ────────────────────────────────────────────
    Elements actRows;
    for (int i = 0; i < static_cast<int>(kActions.size()); ++i) {
      bool sel     = (i == act) && (focus == 1);
      auto numTag  = text(" " + std::to_string(i+1) + " ")
                     | color(Color::Yellow) | bold;
      auto tag     = procs[i].alive
          ? (text(" STOP ") | color(Color::Red)   | bold)
          : (text(" RUN  ") | color(Color::Green) | bold);
      std::string hintStr = (i == MOCK_IDX)
          ? ("word: " + (mockWord.empty() ? "hello" : mockWord))
          : kActions[i].hint;

      auto row = hbox({numTag, tag, text("  "),
                       text(kActions[i].name) | size(WIDTH, EQUAL, 11),
                       text("  "), text(hintStr) | dim});
      actRows.push_back(sel ? (row | inverted) : row);
    }
    auto mockRow = hbox({
        text(" Word: ") | (focus == 2 ? bold : dim),
        mockInput->Render() | size(WIDTH, EQUAL, 24),
        text("  [Tab] to type") | dim,
    });
    auto actionsBox = window(
        text(" Actions   1-6 jump   Enter run·stop ") | bold,
        vbox({vbox(std::move(actRows)), separator(), mockRow}));

    // ── Right middle: test panel ──────────────────────────────────────────
    // 4-column grid of all 12 trained words
    Elements wordRows;
    int nWords = static_cast<int>(kWords.size());
    for (int row = 0; row * WORD_COLS < nWords; ++row) {
      Elements cols;
      for (int c = 0; c < WORD_COLS; ++c) {
        int idx = row * WORD_COLS + c;
        if (idx >= nWords) { cols.push_back(text("") | flex); continue; }
        bool sel = (idx == testWord) && (focus == 3);
        auto numStr = std::to_string(idx + 1);
        auto cell = hbox({
            text(numStr) | color(Color::Yellow) | size(WIDTH, EQUAL, 2),
            text(" "),
            text(kWords[idx]) | size(WIDTH, EQUAL, 6),
        }) | size(WIDTH, EQUAL, 11);
        cols.push_back(sel ? (cell | inverted) : cell);
      }
      wordRows.push_back(hbox(std::move(cols)));
    }
    auto testBox = window(
        text(" Test Model   ↑↓←→ navigate   Enter stream   A all ") | bold,
        vbox(std::move(wordRows)));

    // ── Right bottom: log ─────────────────────────────────────────────────
    Elements logRows;
    {
      std::lock_guard lk(gLogMtx);
      int total   = static_cast<int>(gLog.size());
      // clamp scroll within available range
      logScroll   = std::min(logScroll, std::max(0, total - 1));
      int visible = 12; // rough estimate; FTXUI fills the flex space naturally
      int end     = total - logScroll;
      int start   = std::max(0, end - visible);
      for (int i = start; i < end; ++i) {
        auto &l = gLog[i];
        // Pad source to fixed width so columns align
        std::string srcPad = l.src.substr(0, 12);
        srcPad.resize(12, ' ');
        logRows.push_back(hbox({
            text(l.ts)    | color(Color::GrayDark),
            text(" "),
            text(srcPad)  | color(Color::Cyan),
            text(" "),
            text(l.msg),
        }));
      }
    }
    if (logRows.empty())
      logRows.push_back(text("Waiting for output...") | dim | center);

    auto logBox = window(
        text(" Output   PgUp/PgDn scroll   C clear ") | bold,
        vbox(std::move(logRows))) | flex;

    auto rightCol = vbox({actionsBox, testBox, logBox}) | flex;

    // ── Bottom bar ────────────────────────────────────────────────────────
    auto bar = hbox({
        kd("Tab",      "panel"),
        kd("1-6",      "action"),
        kd("↑↓←→",   "navigate"),
        kd("Enter",    "run/stop"),
        kd("A",        "test all"),
        kd("K",        "kill all"),
        kd("C",        "clear"),
        kd("PgUp/Dn",  "scroll"),
        kd("Q",        "quit"),
    });

    return vbox({
        text(" ★  Inertialink Smart Pen") | bold | center,
        separator(),
        hbox({leftCol, separator(), rightCol}) | flex,
        separator(),
        bar,
    });
  });

  // ── Event handler ──────────────────────────────────────────────────────────
  auto withEvents = CatchEvent(ui, [&](Event ev) -> bool {
    // ── Global: quit ───────────────────────────────────────────────────────
    if (ev == Event::Character('q') || ev == Event::Character('Q')) {
      screen.ExitLoopClosure()();
      return true;
    }

    // ── Global: number keys 1-6 jump to that action ────────────────────────
    // Guard: don't steal keys from text input
    if (focus != 2) {
      for (int i = 0; i < static_cast<int>(kActions.size()); ++i) {
        if (ev == Event::Character(static_cast<char>('1' + i))) {
          act   = i;
          focus = 1;
          syncFt();
          return true;
        }
      }
    }

    // ── Global: kill all ────────────────────────────────────────────────────
    if (ev == Event::Character('k') || ev == Event::Character('K')) {
      for (auto &p : procs) if (p.alive) procStop(p);
      return true;
    }

    // ── Global: clear log (not when typing) ────────────────────────────────
    if (focus != 2 &&
        (ev == Event::Character('c') || ev == Event::Character('C'))) {
      std::lock_guard lk(gLogMtx); gLog.clear(); return true;
    }

    // ── Global: log scroll ─────────────────────────────────────────────────
    if (ev == Event::PageUp)   { logScroll += 5;                    return true; }
    if (ev == Event::PageDown) { logScroll = std::max(0,logScroll-5); return true; }

    // ── Tab: cycle panels ──────────────────────────────────────────────────
    if (ev == Event::Tab) {
      if      (focus == 0)                     focus = 1;
      else if (focus == 1 && act == MOCK_IDX)  focus = 2;
      else if (focus == 1 || focus == 2)        focus = 3;
      else                                      focus = 0;
      syncFt();
      return true;
    }

    // ── Actions panel (focus == 1) ─────────────────────────────────────────
    if (focus == 1) {
      if (ev == Event::ArrowUp) {
        act = std::max(0, act-1); return true;
      }
      if (ev == Event::ArrowDown) {
        act = std::min(static_cast<int>(kActions.size())-1, act+1); return true;
      }
      if (ev == Event::Return) { toggleAction(act); return true; }
    }

    // ── Test panel (focus == 3) ────────────────────────────────────────────
    if (focus == 3) {
      int nWords = static_cast<int>(kWords.size());
      int row    = testWord / WORD_COLS;
      int col    = testWord % WORD_COLS;
      int maxRow = (nWords - 1) / WORD_COLS;

      if (ev == Event::ArrowUp && row > 0) {
        testWord -= WORD_COLS; return true;
      }
      if (ev == Event::ArrowDown && row < maxRow) {
        testWord = std::min(nWords-1, testWord + WORD_COLS); return true;
      }
      if (ev == Event::ArrowLeft && col > 0) {
        testWord -= 1; return true;
      }
      if (ev == Event::ArrowRight && col < WORD_COLS-1 && testWord+1 < nWords) {
        testWord += 1; return true;
      }
      if (ev == Event::Return) {
        streamWord(kWords[testWord]); return true;
      }
      // 'A' — stream all words
      if (ev == Event::Character('a') || ev == Event::Character('A')) {
        streamWord("all"); return true;
      }
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

  logPush("tui", "Ready. 1-6 jump to action, Tab switches panels, Enter runs/stops.");
  screen.Loop(withEvents);

  bgStop = true;
  bgThread.join();
  for (auto &p : procs) if (p.alive) procStop(p);
  return 0;
}
