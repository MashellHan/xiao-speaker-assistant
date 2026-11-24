# Xiao Speaker Assistant — Agent Notes

## Authentication: never access the macOS login Keychain

- All projects and agents must not read, write, unlock or trigger prompts for
  the macOS login Keychain (`login.keychain` / `login.keychain-db`), including
  indirect access through Azure CLI/MSAL, Git/GitHub credential helpers and
  browser password or cookie decryption. Do not retry a path that may access it.
- Explicit environment credentials, private file caches and separately selected
  other keychains may be used for authorized work when the entire path is known
  not to access the login Keychain. Select the path before authentication;
  disable inherited helpers/brokers that may use it, and do not rely on default
  keychain search or fallback behavior. Missing file credentials do not authorize
  fallback.
- Outlook / Microsoft Graph authentication must use explicit file-backed token
  caches (`0600` permissions outside the repository) or explicit environment
  variables—never `msal-extensions` Keychain persistence.
- Never commit personal MAC addresses, serial numbers, Xiaomi tokens, API keys,
  or `.local/` backups into Git.

## Hardware & Flashing Safety (`OH2P`)

- Target hardware: **Xiaomi 智能音箱 Pro (`OH2P`)**.
- **Never cross-flash** an older firmware image (such as `1.58.6`) onto a newer
  `1.62.x+` speaker; always extract/patch from the exact stock firmware version
  running on the device.
- Never write to flash partitions without:
  1. Read-only USB probing and double-read backups (`boot0`, `boot1`, `system0`,
     `system1`) with matching SHA-256 hashes;
  2. Explicit user confirmation accepting brick risk;
  3. Single write to the active `system` partition followed by length-matched
     readback SHA-256 verification before reset.
- If any flash read/write/verify step fails, **stop immediately**—do not retry
  blindly or switch `boot_part` without inspection.

## Browser Verification

- Use Microsoft Edge for local previews and browser-based automation or
  verification when a desktop browser is used.
