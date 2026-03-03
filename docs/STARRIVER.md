# Starriver detector and OCR (Tuanzi Cloud)

BallonsTranslator can use **Starriver Cloud (Tuanzi Manga OCR)** for **text detection** and **OCR**. Detection and OCR run on Starriver’s servers, so no local GPU is needed for those steps.

---

## 1. Get an account

The **dashboard** ([https://dashboard.stariver.org.cn/](https://dashboard.stariver.org.cn/) → [login](https://dashboard.stariver.org.cn/#/user/login)) is for **logging in** and purchasing; it often **does not show a “Register” button**.

Accounts are created via **团子翻译器 (Dango Translator)**:

1. **Download** the Dango Translator app: **[https://translator.dango.cloud/](https://translator.dango.cloud/)** (点击「下载软件」).
2. **Install and open** the app, then open **账号注册** (Account registration) — e.g. from the login window.
3. On the registration form, fill in:
   - **用户名** (Username)
   - **密码** (Password)
   - **手机号** (Phone number) — the number to bind to the account
   - **验证码** (Verification code) — click **获取验证码** (Get verification code) to receive a code (e.g. by SMS), then enter it
4. Click **确定** (Confirm) to complete registration.
5. The **same username and password** then work on the dashboard and in BallonsTranslator (account data is shared: “星河云与团子翻译器的账户数据互通”).

If you cannot install the app or find no register option, consider using a **local detector** instead (e.g. **ctd**, **hf_object_det**, **easyocr_det**, **mmocr_det**) — see Config → Text detector.

---

## 2. Use the Starriver text detector

1. In BallonsTranslator: **Config** → **Text detector** (or **DL Module** → Text detector).
2. Select **stariver_ocr** from the detector dropdown.
3. In the detector params:
   - **User**: your Starriver website **username**.
   - **Password**: your Starriver website **password** (stored in config; avoid on shared PCs).
4. Click **OK** / save config.
5. Run detection as usual (e.g. **Run** on a chapter).  
   The app will log in to Starriver automatically and get a token. On first run (or after changing User/Password), you may see a short delay while the token is fetched.

**Optional:** Use **Update token** to clear the stored token and force a fresh login (e.g. after changing password on the website).

---

## 3. Use the Starriver OCR (optional)

If you want **OCR** to also run on Starriver:

1. **Config** → **OCR** → select **stariver_ocr**.
2. Set the same **User** and **Password** as for the detector.
3. Use **Update token** if you just changed credentials.

You can combine **Starriver detector** with another OCR (e.g. local qwen35), or use **Starriver detector + Starriver OCR** together for a fully cloud-based detect+OCR path.

---

## 4. Detector options (short reference)

| Param | Description |
|-------|-------------|
| **User** / **Password** | Starriver account; required for login. |
| **refine** | Refine detection results (default: on). |
| **filtrate** | Filter results (default: on). |
| **disable_skip_area** | Disable skip-area logic (default: on). |
| **detect_scale** | Detection scale (default: 3). |
| **merge_threshold** | Merge threshold (default: 2.0). |
| **low_accuracy_mode** | Use lower resolution (768px short side) to save time/bandwidth; default off (1536px). |
| **expand_ratio** | Mask expansion ratio (e.g. 0.01). |
| **Update token** | Clear token and log in again. |

---

## 5. If it doesn’t work

- **“Starriver detector token is not set”**  
  Set **User** and **Password** (no placeholders), save config, then run again. Password must be at least 8 characters.

- **“Starriver login failed”**  
  Check username/password at [https://dashboard.stariver.org.cn/#/user/login](https://dashboard.stariver.org.cn/#/user/login) and that the site is reachable (no firewall/VPN blocking).

- **“Starriver detector request failed”**  
  Network or server issue. Retry later; if it persists, check the Starriver site/status or try **Update token**.

- **Proxy**  
  If you need a proxy for the app’s HTTP requests, configure it in Config (e.g. under General or the module that uses `requests`). Starriver uses `requests.post` to Starriver APIs.

---

## URLs used by the app

- **Register (create account):** Usually via the **团子翻译器 (Dango Translator)** app — [https://translator.dango.cloud/](https://translator.dango.cloud/) (download, then register inside the app). The dashboard may not offer a web register button.
- **Dashboard / login (browser):** [https://dashboard.stariver.org.cn/](https://dashboard.stariver.org.cn/) → [https://dashboard.stariver.org.cn/#/user/login](https://dashboard.stariver.org.cn/#/user/login)
- **Login API (used by BallonsTranslator):** `https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login`
- **Detect/OCR API:** `https://dl.ap-qz.starivercs.cn/v2/manga_trans/advanced/manga_ocr`

Use the Dango Translator app to create an account; then use the same username and password in BallonsTranslator.
