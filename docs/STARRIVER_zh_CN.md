# Starriver 检测器与 OCR（团子云）

BallonsTranslator 支持使用 **Starriver Cloud（团子漫画 OCR）** 进行文本检测与 OCR。相关步骤在云端完成，对本地 GPU 依赖较低。

---

## 1）账号获取

星河云后台（<https://dashboard.stariver.org.cn/>）主要用于登录/购买，很多时候没有明显“注册”按钮。  
通常需要通过 **团子翻译器** 客户端注册：

1. 下载：<https://translator.dango.cloud/>（点击“下载软件”）
2. 安装后打开，在登录页进入“账号注册”
3. 填写：用户名、密码、手机号、验证码（点“获取验证码”）
4. 点击“确定”完成注册

注册后同一套账号可在星河云后台与 BallonsTranslator 使用（账户数据互通）。

---

## 2）使用 Starriver 文本检测

1. 打开 **Config → Text detector**（或 **DL Module → Text detector**）
2. 选择 `stariver_ocr`
3. 参数里填写：
   - `User`：星河云用户名
   - `Password`：星河云密码
4. 保存后正常运行检测（Run）

首次登录（或改密码后）会有短暂 token 获取延迟。

可选：点击 **Update token** 清空旧 token 并强制重新登录。

---

## 3）使用 Starriver OCR（可选）

若 OCR 也想走云端：

1. **Config → OCR** 选择 `stariver_ocr`
2. 填同样的 `User` / `Password`
3. 若刚改账号信息，执行 `Update token`

可组合方式：
- `Starriver detector + 本地 OCR`
- `Starriver detector + Starriver OCR`（全云端）

---

## 4）关键参数速览

| 参数 | 说明 |
|---|---|
| User / Password | 登录必填 |
| refine | 结果细化（默认开） |
| filtrate | 过滤结果（默认开） |
| disable_skip_area | 关闭跳过区域逻辑（默认开） |
| detect_scale | 检测尺度（默认 3） |
| merge_threshold | 合并阈值（默认 2.0） |
| low_accuracy_mode | 低分辨率快速模式（默认关） |
| expand_ratio | mask 扩张比例 |
| Update token | 清 token 并重新登录 |

---

## 5）常见问题

- **“Starriver detector token is not set”**
  - 未填写或未保存账号密码；密码需至少 8 位。
- **“Starriver login failed”**
  - 账号密码错误，或后台站点不可达（网络/代理/防火墙）。
- **“Starriver detector request failed”**
  - 网络或服务端异常，稍后重试；必要时点 Update token。

---

## 本项目使用到的地址

- 后台登录：<https://dashboard.stariver.org.cn/#/user/login>
- 团子翻译器下载（注册入口）：<https://translator.dango.cloud/>
- 登录 API：`https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login`
- Detect/OCR API：`https://dl.ap-qz.starivercs.cn/v2/manga_trans/advanced/manga_ocr`

