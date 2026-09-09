// ============================================================
// 九宫格赛道自动计分系统 - 正式版（带OLED，串口动态布局）
// 映射：卡槽0~5→位置0~5，卡槽6→位置7，卡槽7→位置8，左下角（位置6）始终为空
// 位置6（左下）固定为空，首次触发开始并计分，15分钟超时
// OLED：SDA=GPIO20，SCL=GPIO21
// ============================================================

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <WiFi.h>        // 用于连接WiFi
#include <HTTPClient.h>  // 用于发送HTTP请求
// ============================================================
// 一、OLED配置
// ============================================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ============================================================
// 二、默认布局（串口未输入时使用）
// ============================================================
const int DEFAULT_LAYOUT[8] = {6, 5, 4, 7, 2, 3, 1, 8};

// ============================================================
// 三、引脚定义
// ============================================================
const int SW_ROWS = 3;
const int SW_COLS = 3;
const int swRowPins[SW_ROWS] = {11, 12, 13};
const int swColPins[SW_COLS] = {14, 15, 16};

// ============================================================
// 四、配置参数
// ============================================================
const int SLOT_COUNT = 8;
// 卡槽0~5→位置0~5，卡槽6→位置7，卡槽7→位置8
const int slotToGrid[SLOT_COUNT] = {0, 1, 2, 3, 4, 5, 7, 8};
// 微动开关物理位置→卡槽映射
const int physicalToSlot[9] = {0, 1, 2, 3, 4, 5, -1, 6, 7};

const int TARGET_SEQUENCE[7] = {1, 2, 3, 4, 5, 6, 7};
const int TARGET_COUNT = 7;
const float SCORE_PER_TARGET = 10.0;
const unsigned long TIMEOUT_MS = 900000;  // 15分钟
const int PANEL_ENTRY = 1;
const int PANEL_EXIT = 7;
const unsigned long DEBOUNCE_TIME = 200;
const unsigned long SERIAL_TIMEOUT = 60000; // 60秒

// ============================================================
// 五、全局变量
// ============================================================
int slotToPanel[SLOT_COUNT] = {0};
bool scored[TARGET_COUNT] = {false};
int targetIndex = 0;
float taskScore = 0.0;
bool roundComplete = false;
bool roundStarted = false;
bool timedOut = false;
unsigned long startTime = 0;
unsigned long elapsedSeconds = 0;
unsigned long lastTriggerTime[9] = {0};
  // 每个卡槽的上次触发时间（用于冷却）
unsigned long lastTriggerCooldown[SLOT_COUNT] = {0};
const unsigned long TRIGGER_COOLDOWN = 2000;  // 冷却时间 2000 毫秒（2 秒）

// ============================================================
// WiFi与服务器配置（请修改为你的实际信息）
// ============================================================
const char* WIFI_SSID = "eelab901b_2";       // 替换为实际的WiFi名称
const char* WIFI_PASSWORD = "901901901";   // 替换为实际的WiFi密码
const char* SERVER_URL = "http://192.168.31.254:8001/api/v1/scores"; // 替换为你的服务器地址
// ============================================================
// 六、OLED显示函数
// ============================================================
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.println("Nine-Grid Race");
  display.println("------------");

  if (!roundStarted) {
    display.println("Status: WAITING");
    display.println("Press any switch");
  } else if (roundComplete) {
    display.println("Status: COMPLETE!");
  } else {
    display.println("Status: RUNNING");
  }

  display.print("Score: ");
  display.println(taskScore, 1);

  display.print("Progress: ");
  for (int i = 0; i < TARGET_COUNT; i++) {
    if (i < targetIndex) display.print(TARGET_SEQUENCE[i]);
    else if (i == targetIndex) { display.print("["); display.print(TARGET_SEQUENCE[i]); display.print("]"); }
    else display.print(TARGET_SEQUENCE[i]);
    if (i < TARGET_COUNT - 1) display.print("→");
  }
  display.println();

  if (roundStarted) {
    display.print("Time: ");
    display.print(elapsedSeconds);
    display.println("s");
  }
  display.display();
}

void displayStartup() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.println("Nine-Grid System");
  display.println("Send layout via serial");
  display.println("e.g. 6,5,4,3,2,1,7,8");
  display.display();
  delay(2000);
}

void displayComplete() {
  float finalTaskScore = taskScore > 70 ? 70 : taskScore;
  int completionScore = (roundComplete && !timedOut) ? 30 : 0;
  float totalScore = finalTaskScore + completionScore;
  if (totalScore > 100) totalScore = 100;

  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(2);
  display.println("COMPLETE!");
  display.setTextSize(1);
  display.print("Task: ");
  display.println(finalTaskScore, 1);
  display.print("Comp: ");
  display.print(completionScore);
  display.print("  Total: ");
  display.println(totalScore, 1);
  display.print("Time: ");
  display.print(elapsedSeconds);
  display.println("s");
  display.display();
}

// ============================================================
// 七、微动开关扫描
// ============================================================
int scanSwitches() {
  for (int col = 0; col < SW_COLS; col++) {
    digitalWrite(swColPins[col], LOW);
    for (int row = 0; row < SW_ROWS; row++) {
      if (digitalRead(swRowPins[row]) == LOW) {
        int physicalPos = row * SW_COLS + col;
        if (physicalPos >= 9) { digitalWrite(swColPins[col], HIGH); continue; }
        unsigned long now = millis();
        if (now - lastTriggerTime[physicalPos] > DEBOUNCE_TIME) {
          lastTriggerTime[physicalPos] = now;
          digitalWrite(swColPins[col], HIGH);
          if (physicalPos < 0 || physicalPos > 8) return -1;
          int realSlot = physicalToSlot[physicalPos];
          if (realSlot < 0) return -1;
          return realSlot;
        }
      }
    }
    digitalWrite(swColPins[col], HIGH);
  }
  return -1;
}

// ============================================================
// 八、计分逻辑
// ============================================================
void processTrigger(int slotIndex) {
  // ----- 冷却检查 -----
  unsigned long now = millis();
  if (now - lastTriggerCooldown[slotIndex] < TRIGGER_COOLDOWN) {
    return;
  }
  lastTriggerCooldown[slotIndex] = now;
  // ----- 冷却检查结束 -----

  if (!roundStarted || roundComplete) return;
  int panelID = slotToPanel[slotIndex];
  if (panelID == 0) {
    Serial.printf("卡槽%d无面板\n", slotIndex);
    return;
  }
  int gridPos = slotToGrid[slotIndex];
  Serial.printf("\n📍 卡槽%d (位置%d) → 面板%d\n", slotIndex, gridPos, panelID);

  for (int i = 0; i < TARGET_COUNT; i++) {
    if (TARGET_SEQUENCE[i] == panelID) {
      if (i == targetIndex && !scored[i]) {
        // ========== 顺序正确，计分 ==========
        taskScore += SCORE_PER_TARGET;
        scored[i] = true;
        targetIndex++;

        if (panelID == PANEL_ENTRY) {
          Serial.printf("🚪 入口1! +%.1f分\n", SCORE_PER_TARGET);
        } else if (panelID == PANEL_EXIT) {
          Serial.printf("🏁 到达面板7! +%.1f分\n", SCORE_PER_TARGET);
        } else {
          Serial.printf("✅ 面板%d! +%.1f分\n", panelID, SCORE_PER_TARGET);
        }
        Serial.printf("📊 当前得分: %.1f / 70\n", taskScore);

        // ========== 新逻辑：根据情况决定是否结束 ==========
        if (panelID == PANEL_EXIT) {
          // 踩到面板7：只有走完1-6才结束，否则继续
          if (targetIndex >= TARGET_COUNT) {
            // 全部7个目标都已完成 → 结束
            roundComplete = true;
            elapsedSeconds = (millis() - startTime) / 1000;
            Serial.println("\n🎉 赛道完成!");
            displayComplete();
            reportFinalScore();
          } else {
            // 提前踩到面板7，展示当前进度后继续
            Serial.printf("⏳ 面板7已踩，但还需完成面板%d\n", TARGET_SEQUENCE[targetIndex]);
          }
        } else {
          // 踩到面板1-6：正常计分，检查是否完成全部
          if (targetIndex >= TARGET_COUNT) {
            roundComplete = true;
            elapsedSeconds = (millis() - startTime) / 1000;
            Serial.println("\n🎉 赛道完成!");
            displayComplete();
            reportFinalScore();
          }
        }

        if (targetIndex < TARGET_COUNT) {
          Serial.printf("🎯 下一个目标：面板%d\n", TARGET_SEQUENCE[targetIndex]);
        }

        updateDisplay();
        return;

      } else if (scored[i]) {
        Serial.printf("⚠️ 重复面板%d\n", panelID);
      } else {
        Serial.printf("❌ 顺序错误！期望%d, 实际%d\n", TARGET_SEQUENCE[targetIndex], panelID);
      }
      updateDisplay();
      return;
    }
  }
  Serial.printf("⏳ 非计分面板%d\n", panelID);
}

// ============================================================
// 九、成绩报告
// ============================================================
void reportFinalScore() {
  float finalTaskScore = taskScore > 70 ? 70 : taskScore;
  int completionScore = (roundComplete && !timedOut) ? 30 : 0;
  float totalScore = finalTaskScore + completionScore;
  if (totalScore > 100) totalScore = 100;
  Serial.println("\n========================================");
  Serial.println("  📊 最终成绩报告");
  Serial.println("========================================");
  Serial.printf("  🏆 任务完成得分：%.1f / 70\n", finalTaskScore);
  Serial.printf("  ⏱️  完成分：%d / 30\n", completionScore);
  Serial.printf("  📊 最终总分：%.1f / 100\n", totalScore);
  Serial.printf("  ✅ 正确步数：%d / %d\n", targetIndex, TARGET_COUNT);
  Serial.printf("  ⏱️  用时：%lu 秒\n", elapsedSeconds);
  Serial.printf("  📌 状态：%s\n", roundComplete ? (timedOut ? "超时 ❌" : "已完成 ✅") : "未完成 ❌");
  Serial.println("========================================\n");
  String json = "{";
  json += "\"module_id\":3,\"robot_id\":15,\"device_ip\":171,\"event\":\"FINISH\",";
  json += "\"timestamp\":" + String(millis() / 1000) + ",";
  json += "\"penalty\":0,";
  json += "\"score\":" + String(totalScore) + "}";
  
  Serial.println("📤 JSON成绩：");
  Serial.println(json);
  uploadScoreToServer(json);
}

// ============================================================
// 通过WiFi上传成绩到服务器
// ============================================================
bool uploadScoreToServer(String jsonData) {

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi未连接，无法上传");
        return false;
    }

    HTTPClient http;

    bool beginResult = http.begin(SERVER_URL);

    Serial.printf("WiFi状态: %d\n", WiFi.status());
    Serial.printf("SERVER_URL: %s\n", SERVER_URL);
    Serial.printf("http.begin(): %d\n", beginResult);

    if (!beginResult) {
        Serial.println("❌ HTTPClient初始化失败");
        return false;
    }

    http.addHeader("Content-Type", "application/json");

    Serial.println("📤 正在上传成绩...");

    int httpCode = http.POST(jsonData);

    Serial.printf("http.POST() 返回: %d\n", httpCode);

    if (httpCode > 0) {

        Serial.printf("服务器 HTTP 状态码: %d\n", httpCode);

        if (httpCode == HTTP_CODE_OK) {
            Serial.println("✅ 成绩上传成功！");
        } else {
            Serial.printf("⚠️ 服务器响应异常，HTTP代码: %d\n", httpCode);
        }

    } else {

        Serial.printf("❌ 上传失败，错误代码: %d\n", httpCode);
        Serial.printf("HTTPClient错误: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();

    return httpCode == HTTP_CODE_OK;
}
// ============================================================
// 十、串口输入解析（超时60秒）
// ============================================================
void readLayoutFromSerial() {
  Serial.println("\n⏳ 等待串口输入布局数据...");
  Serial.println("格式：8个数字，用逗号或空格分隔，例如：6,5,4,3,2,1,7,8");
  Serial.println("（卡槽0~5→位置0~5，卡槽6→位置7，卡槽7→位置8，位置6固定为空）");
  Serial.println("请在60秒内输入，超时将使用默认布局。");

  unsigned long startWait = millis();
  String input = "";
  while (millis() - startWait < SERIAL_TIMEOUT) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') {
        if (input.length() > 0) break;
      } else {
        input += c;
      }
    }
    delay(10);
  }
  if (input.length() == 0) {
    Serial.println("⏰ 未收到输入，使用默认布局。");
    for (int i = 0; i < SLOT_COUNT; i++) slotToPanel[i] = DEFAULT_LAYOUT[i];
    return;
  }
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == ',' || input[i] == ';') input[i] = ' ';
  }
  int values[8];
  int count = 0;
  char *ptr = strtok((char*)input.c_str(), " ");
  while (ptr != NULL && count < 8) {
    values[count++] = atoi(ptr);
    ptr = strtok(NULL, " ");
  }
  if (count == 8) {
    for (int i = 0; i < 8; i++) slotToPanel[i] = values[i];
    Serial.println("✅ 布局已更新！");
  } else {
    Serial.println("⚠️ 解析失败，使用默认布局。");
    for (int i = 0; i < SLOT_COUNT; i++) slotToPanel[i] = DEFAULT_LAYOUT[i];
  }
}

// ============================================================
// 十一、打印九宫格布局（位置6固定空）
// ============================================================
void printLayout() {
  Serial.println("  当前面板布局（左下角位置6固定为空）：");
  Serial.println("  ┌─────────┬─────────┬─────────┐");
  for (int r = 0; r < 3; r++) {
    Serial.print("  │");
    for (int c = 0; c < 3; c++) {
      int pos = r * 3 + c;
      int panel = 0;
      for (int s = 0; s < SLOT_COUNT; s++) {
        if (slotToGrid[s] == pos) {
          panel = slotToPanel[s];
          break;
        }
      }
      if (pos == 6) {
        Serial.print(" 空 ");
      } else if (panel > 0) {
        Serial.printf("  %d  ", panel);
      } else {
        Serial.print(" 空 ");
      }
      Serial.print("│");
    }
    Serial.println();
    if (r < 2) Serial.println("  ├─────────┼─────────┼─────────┤");
  }
  Serial.println("  └─────────┴─────────┴─────────┘\n");
}

void connectWiFi() {
  Serial.print("正在连接WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi连接成功！");
    Serial.print("IP地址: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ WiFi连接失败，请检查账号密码");
  }
}

// ============================================================
// 十二、初始化
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  connectWiFi();

  Wire.begin(20, 21);   // SDA=GPIO20, SCL=GPIO21
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("❌ OLED初始化失败");
  }

  // 初始化微动引脚
  for (int r = 0; r < SW_ROWS; r++) pinMode(swRowPins[r], INPUT_PULLUP);
  for (int c = 0; c < SW_COLS; c++) {
    pinMode(swColPins[c], OUTPUT);
    digitalWrite(swColPins[c], HIGH);
  }

  Serial.println("\n========================================");
  Serial.println("  九宫格赛道手动计分系统（正式版）");
  Serial.println("  规则: 1→2→3→4→5→6→7 (每个10分)");
  Serial.println("  首次触发开始并计分，15分钟超时");
  Serial.println("  带OLED显示，串口动态布局");
  Serial.println("========================================\n");

  // 串口输入布局
  readLayoutFromSerial();
  printLayout();

  displayStartup();
  updateDisplay();

  Serial.println("✅ 系统就绪，首次触发微动开关将开始计时并计分");
  Serial.println("========================================\n");
}

// ============================================================
// 十三、主循环
// ============================================================
void loop() {
  if (!roundStarted && !roundComplete) {
    int slot = scanSwitches();
    if (slot != -1) {
      roundStarted = true;
      startTime = millis();
      Serial.println("🚀 比赛开始！");
      processTrigger(slot);
      updateDisplay();
    }
  }

  if (roundStarted && !roundComplete) {
    elapsedSeconds = (millis() - startTime) / 1000;
    if (millis() - startTime > TIMEOUT_MS) {
      timedOut = true;
      roundComplete = true;
      elapsedSeconds = (millis() - startTime) / 1000;
      Serial.println("\n⏰ 比赛超时（15分钟）！");
      reportFinalScore();
      displayComplete();
      updateDisplay();
      return;
    }
    int slot = scanSwitches();
    if (slot != -1) processTrigger(slot);

    static unsigned long lastDisp = 0;
    if (millis() - lastDisp > 1000) {
      lastDisp = millis();
      updateDisplay();
    }
  }

  if (!roundStarted && !roundComplete) {
    static unsigned long lastIdle = 0;
    if (millis() - lastIdle > 500) {
      lastIdle = millis();
      updateDisplay();
    }
  }
  delay(10);
}