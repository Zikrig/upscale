# Video pipeline (RTX 3090 / 32 GB RAM / 160 GB disk)

Инстанс: `rtx3090-1.8.32.160` — **8 vCPU, 32 GB RAM, 160 GB disk, 1× RTX 3090**.

Сервер: `ubuntu@195.209.214.89`  
Ключ (на ПК): `C:\Users\Ф\Desktop\projects\upscale\ttttest-185642-zigrik.pem`

Локальные видео:
`C:\Users\Ф\Desktop\фильмы\FINISHED\vibecoder\finale\to_translate\`

**Важно:** команды `scp` / PowerShell запускайте **на своём ПК**, не внутри SSH (`ubuntu@ttttest`).

Ограничения:
- без клона голоса;
- один ролик за раз на GPU;
- диск 160 GB — после скачивания результатов удаляйте лишнее с сервера;
- `to_translate_p4_not_face.mp4` — на этапе lip-sync пропускается.

---

## Этап 1 — субтитры (сейчас)

Выход:
- `output/subs/*.srt` + `*.txt`
- `output/subs_burned/*_subs.mp4` (burn-in)

Фразы: ~2.0–2.5 с, до ~42 символов.

### 1. Загрузить код на сервер (если ещё нет git)

```powershell
$pem = "C:\Users\Ф\Desktop\projects\upscale\ttttest-185642-zigrik.pem"
# предпочтительно: git push с ПК, на сервере git pull
```

На сервере:

```bash
cd ~/upscale
git pull
chmod +x subs.sh
```

### 2. Загрузить MP4 на сервер (с ПК)

```powershell
$pem = "C:\Users\Ф\Desktop\projects\upscale\ttttest-185642-zigrik.pem"
$src = "C:\Users\Ф\Desktop\фильмы\FINISHED\vibecoder\finale\to_translate"
ssh -i $pem ubuntu@195.209.214.89 "mkdir -p ~/upscale/video_input"

# все файлы
scp -i $pem -o ServerAliveInterval=30 `
  "$src\*.mp4" `
  ubuntu@195.209.214.89:~/upscale/video_input/

# или по одному (если диск/сеть узкие)
# scp -i $pem "$src\to_translate_p1.mp4" ubuntu@195.209.214.89:~/upscale/video_input/
```

Проверка на сервере:

```bash
ls -lh ~/upscale/video_input/
df -h /
```

### 3. Поставить зависимости (если нужно)

```bash
cd ~/upscale
source .venv/bin/activate
pip install faster-whisper
sudo apt install -y ffmpeg fonts-dejavu-core
```

### 4. Запуск субтитров

```bash
cd ~/upscale
source .venv/bin/activate
bash subs.sh

# один файл
bash subs.sh --only p1 --force

# пересобрать всё
bash subs.sh --force
```

Первый запуск скачает Whisper `large-v3` в `~/upscale/models/whisper`.

### 5. Скачать результат на ПК

```powershell
$pem = "C:\Users\Ф\Desktop\projects\upscale\ttttest-185642-zigrik.pem"
$dest = "C:\Users\Ф\Desktop\projects\upscale\output"
New-Item -ItemType Directory -Force -Path "$dest\subs", "$dest\subs_burned" | Out-Null

scp -i $pem -r ubuntu@195.209.214.89:~/upscale/output/subs/. "$dest\subs\"
scp -i $pem -o ServerAliveInterval=30 -r `
  ubuntu@195.209.214.89:~/upscale/output/subs_burned/. `
  "$dest\subs_burned\"
```

### 6. Освободить диск на сервере

```bash
df -h /
rm -rf /tmp/subs_*
# после успешного скачивания burned-роликов:
# rm -f ~/upscale/output/subs_burned/*.mp4
# при необходимости исходники:
# rm -f ~/upscale/video_input/to_translate_p1.mp4
```

---

## Этап 2 — EN-аудио (задел, без клона)

План:
1. Перевести текст сегментов SRT на английский (или Whisper `task=translate`).
2. Озвучить **edge-tts** (нейтральный EN-голос, без клона).
3. Уложить каждый кусок **в то же временное окно**, что у SRT (лёгкий trim/pad), чтобы длина = исходному видео.

Хронометраж видео **не меняется** (нужно для lip-sync).

Команды появятся после реализации модуля (черновик):

```bash
# будущий скрипт
# bash dub_en.sh --only p1 --force
```

---

## Этап 3 — Lip-sync (задел)

План: **Wav2Lip** (или аналог) — лицо под новую EN-дорожку, по одному ролику.

- Вход: исходный MP4 + EN WAV той же длительности.
- Выход: `output/lipsync/*_en_lipsync.mp4`
- Пропуск: `*_not_face*`

VRAM/диск: не держать несколько моделей сразу; чистить temp после каждого ролика.

```bash
# будущий скрипт
# bash lipsync.sh --only p1 --force
```

---

## Быстрый чеклист этапа 1

| Шаг | Где | Команда |
|-----|-----|---------|
| Upload MP4 | ПК PowerShell | `scp … video_input/` |
| Subs | сервер | `bash subs.sh` |
| Download | ПК PowerShell | `scp … output/subs` и `subs_burned` |
| Cleanup | сервер | `rm -rf /tmp/subs_*` + лишние mp4 |

Параметры длины фраз: `SUB_TARGET_SEC`, `SUB_MAX_CHARS` в `app/config.py`.
