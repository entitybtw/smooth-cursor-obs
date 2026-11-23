"""Рендер: ffmpeg накладывает PNG-спрайты курсора через overlay, позиции покадрово задаёт sendcmd.

Каждый спрайт (тип курсора × шаг анимации клика, «призраки» шлейфа, отладочный сырой курсор) — отдельный overlay.
Неактивные спрятаны за кадром (x = HIDE), поэтому смена типа — это просто перенос двух overlay.
"""
import csv
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

import smoothing
import platform_cursor as wc
from i18n import t

log = logging.getLogger("smooth")
HIDE = -10000
CLICK_STEPS = 16  # шагов анимации клика (0 — обычный курсор)
TILT_STEP_DEG = 1.0  # шаг угла наклона в движении: поворот плавный, ступенек не видно, а спрайтов немного
TILT_SPEED = 0.6  # ширин кадра в секунду по горизонтали: на этой скорости поворот — 3/4 от полного
TILT_SMOOTH_MS = 70  # сглаживание угла по времени: поворачивается плавно, без рывков от толчков руки
CLICK_TILT_STEP = 4  # во время анимации клика наклон — шагами по 4 шага: пар «клик × наклон» вчетверо меньше
MORPH_STEPS = 16  # шагов анимации смены вида курсора (0 — без смены): отскоку нужны промежуточные размеры
MORPH_OUT = 0.3  # доля смены вида, за которую прежний курсор сжимается в точку; остальное — рост нового


def click_curve(age_ms, duration_ms, press_ms=0):
    """Сила анимации клика 0..1..0: за первую четверть длительности вжимается (ease-out), держится, пока кнопка
    зажата (press_ms — сколько её держали, inf — ещё держат), потом за оставшиеся 3/4 плавно отпускает."""
    attack, release = 0.25 * duration_ms, 0.75 * duration_ms
    hold = np.maximum(0, np.asarray(press_ms, float) - attack)
    age = np.asarray(age_ms, float)
    with np.errstate(invalid="ignore"):  # inf − inf у «давно отпущенных» → nan → считаем отпущенными
        v = np.nan_to_num(np.clip((age - attack - hold) / release, 0, 1), nan=1.0)
    rise = 1 - (1 - np.clip(age / attack, 0, 1)) ** 3  # до первого клика age < 0 → 0
    return np.where(age < attack, rise, np.where(age < attack + hold, 1.0, 1 - v * v * (3 - 2 * v)))


def click_press(tf, downs, lens, duration_ms):
    """Сила анимации клика по кадрам (моменты tf, мс): рендер знает клики заранее, поэтому курсор начинает вжиматься
    до нажатия, и сильнее всего он сжат и наклонён ровно в момент клика. Отпускать начинает, когда отпустили кнопку.
    downs — моменты нажатий (по возрастанию), lens — сколько держали (inf — не отпустили)."""
    lead = 0.25 * duration_ms  # вжимание у click_curve — первая четверть длительности
    last = np.maximum(np.searchsorted(downs, tf + lead, "right") - 1, 0)
    return click_curve(tf + lead - downs[last], duration_ms, lens[last] + lead)


def press_sprite(img, hot, scale, deg):
    """Сжатие и поворот против часовой вокруг hotspot: точка клика остаётся на месте. Возвращает (картинка, hotspot)."""
    x0, y0, x1, y1 = img.getbbox() or (0, 0, *img.size)
    # холст — круг до дальнего угла курсора: повёрнутый не обрежется, а лишних пикселей не крутим (это дорого).
    # Кратно 4: при уменьшении 4× спрайта пиксели ложатся на сетку обычного курсора, а не между ними (мыло).
    r = 4 * math.ceil((max(math.hypot(x - hot[0], y - hot[1]) for x in (x0, x1) for y in (y0, y1)) + 2) / 4)
    c = Image.new("RGBa", (2 * r, 2 * r))
    c.paste(img.convert("RGBa"), (round(r - hot[0]), round(r - hot[1])))
    n = round(2 * r * scale)
    c = c.rotate(deg, Image.BICUBIC, center=(r, r)).resize((n, n), Image.LANCZOS).convert("RGBA")
    box = c.getbbox()
    if box is None:  # пустой курсор программы или сжат до невидимости — прозрачная точка, как в morph_sprite
        return Image.new("RGBA", (1, 1)), (0, 0)
    return c.crop(box), (n / 2 - box[0], n / 2 - box[1])


def pose_sprite(big, big_hot, rc, lvl, tilt):
    """Курсор из 4× спрайта: шаг анимации клика lvl и наклон в движении tilt (в шагах TILT_STEP_DEG).
    Возвращает (картинка, hotspot)."""
    p = lvl / (CLICK_STEPS - 1)
    return press_sprite(big, big_hot, (1 - (1 - rc["click_scale"]) * p) / 4,
                        rc["click_tilt_deg"] * p + tilt * TILT_STEP_DEG)


@functools.lru_cache
def _bump(amount):
    """Сила горба отскока, при которой рост проскакивает ровно на amount (подбор делением пополам)."""
    u = np.linspace(0, 1, 401)
    base, bump = 1 - (1 - u) ** 3, np.sin(np.pi * u) ** 2 * u
    lo, hi = 0.0, 5.0
    for _ in range(40):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if (base + k * bump).max() - 1 < amount else (lo, k)
    return lo


def bounce_curve(u, amount):
    """Рост нового вида 0 → 1 за u 0..1: ease-out и горб отскока — проскакивает на amount (0.12 — на 12% крупнее)
    и мягко возвращается; 0 — без отскока. В конце ровно 1 и без рывка."""
    return 1 - (1 - u) ** 3 + _bump(round(float(amount), 3)) * math.sin(math.pi * u) ** 2 * u


def morph_sprite(old, old_hot, new, new_hot, p, bounce=0.12, k=0.25, deg=0):
    """Смена вида курсора, p 0..1: за первые MORPH_OUT прежний сжимается в точку (к своему центру, с ускорением),
    затем новый плавно вырастает из точки с одним мягким отскоком — проскакивает на bounce крупнее и возвращается. old/new — спрайты в 1/k размера (по умолчанию 4×, как для анимации клика),
    результат — в обычном. Точка клика (hotspot) нового — на месте. deg — наклон в движении, вокруг hotspot, как у
    pose_sprite. Возвращает (картинка, hotspot)."""
    if p < MORPH_OUT:
        img, hot, scale = old, old_hot, 1 - (p / MORPH_OUT) ** 2
    else:
        u = (p - MORPH_OUT) / (1 - MORPH_OUT)
        img, hot, scale = new, new_hot, bounce_curve(u, bounce)
    if scale < 0.08:  # момент между ними: курсор — точка
        return Image.new("RGBA", (1, 1)), (0, 0)
    x0, y0, x1, y1 = img.getbbox() or (0, 0, *img.size)
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    im, (cx, cy) = press_sprite(img, c, scale * k, deg)
    dx, dy, a = (c[0] - hot[0]) * k, (c[1] - hot[1]) * k, math.radians(deg)
    # центр — там же, где у несжатого курсора, повёрнутого на deg вокруг hotspot
    return im, (cx - dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) - dy * math.cos(a))


def morph_steps(kind, vis, fps, ms):
    """Шаг смены вида курсора по кадрам (0 — нет смены) и прежний вид. Смена начинается с первого кадра нового вида
    и длится ms; сменился снова посреди анимации — новая идёт от вида, что был до неё. После скрытого курсора
    (печатают текст) — без анимации: появляется сразу нужным."""
    step, prev = np.zeros(len(kind), int), np.asarray(kind).copy()
    dur, last, t0, frm = ms * fps / 1000, None, None, None
    for n in range(len(kind)):
        if not vis[n]:
            last = t0 = None
            continue
        if last is not None and kind[n] != last:
            t0, frm = n, last
        last = kind[n]
        if t0 is not None:
            p = (n - t0 + 1) / (dur + 1)
            if p >= 1:
                t0 = None
                continue
            step[n], prev[n] = min(MORPH_STEPS - 1, max(1, round(p * MORPH_STEPS))), frm
    return step, prev


def hold_types(kind, vis, ms, fps):
    """Вид курсора, мелькнувший короче ms (пролетели кнопку), не показываем: остаётся прежний. После скрытого курсора
    и в конце видео — как есть."""
    out, kind = np.asarray(kind).copy(), np.asarray(kind)
    need, cur, i = ms * fps / 1000, None, 0
    while i < len(out):
        j = i + 1
        while j < len(out) and vis[j] == vis[i] and kind[j] == kind[i]:
            j += 1
        if not vis[i]:
            cur = None
        elif cur is not None and kind[i] != cur and j - i < need and j < len(out):
            out[i:j] = cur
        else:
            cur = kind[i]
        i = j
    return out


def tilt_angle(v, right_deg, left_deg):
    """Угол наклона в движении, ° (плюс — против часовой), по скорости v (ширин кадра в секунду по горизонтали, плюс —
    вправо). В покое курсор как в системе; на ходу поворачивается от этого положения в сторону движения: вправо — на
    right_deg, влево — на left_deg (у стрелки остриё скошено на 22.5°: при повороте на 22° вправо она смотрит прямо
    вверх). Минус — назад, как от инерции: едет вправо — поворот влево. Поворот растёт со скоростью: медленно — чуть-чуть,
    быстро — полный. Во время анимации клика наклон тоже есть: поворот клика ложится поверх."""
    f = np.tanh(np.asarray(v) / TILT_SPEED)
    return -f * np.where(f > 0, right_deg, left_deg)  # f: −1…1 — насколько полно повернуть и в какую сторону


def smooth_tilt(deg, fps):
    """Угол по кадрам, сглаженный по времени гауссом (σ = TILT_SMOOTH_MS): без запаздывания, как траектория."""
    s = TILT_SMOOTH_MS * fps / 1000
    h = math.ceil(3 * s)
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
    return np.convolve(deg, k / k.sum())[h:h + len(deg)]


def frame_tilt(v, ok, fps, right_deg, left_deg, step=TILT_STEP_DEG):
    """Наклон по кадрам, в шагах step (None — в градусах, без округления): tilt_angle по скорости v (там, где не ok,
    — 0), сглаженный по времени. И во время анимации клика: раньше там наклона не было, и движение сразу после клика
    проходило без него."""
    deg = smooth_tilt(np.where(ok, tilt_angle(v, right_deg, left_deg), 0.0), fps)
    return np.round(deg / step).astype(int) if step else deg


def click_tilt(tilt, level):
    """Шаг наклона с учётом клика: на кадрах его анимации — кратный CLICK_TILT_STEP (как у шлейфа), иначе пары
    «шаг клика × шаг наклона» дали бы сотни спрайтов. Разница в пару градусов на сжатом курсоре не видна."""
    tilt = np.asarray(tilt)
    return np.where(np.asarray(level) > 0, np.round(tilt / CLICK_TILT_STEP).astype(int) * CLICK_TILT_STEP, tilt)


def ghost_tilt(k):
    """Шаг наклона «призраков» шлейфа — каждый четвёртый: они бледные и сдвинуты назад, разницы в пару градусов не
    видно, а слоёв в ffmpeg вчетверо меньше (каждый слой — время на каждом кадре; при наклоне 72/27 рендер в 1.3×
    быстрее, чем с шагом 2)."""
    return int(np.round(k / 4)) * 4


def tilt_tag(k):
    """Суффикс имени спрайта с наклоном: _m3 — по часовой на 3 шага, _p2 — против часовой."""
    return f"_{'p' if k > 0 else 'm'}{abs(k)}" if k else ""


def ghost_alphas(rc):
    """Непрозрачность «призраков» шлейфа от дальнего к ближнему; чем длиннее шлейф, тем их больше."""
    n = int(np.clip(round(12 * rc["blur_length"]), 2, 16))
    return [min(1.0, 0.45 * rc["blur_opacity"] * (i + 1) / n) for i in range(n)]


APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # только на Windows


FFMPEG_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SmoothCursor" / "ffmpeg"  # сюда ставит программа
EXE = ".exe" if sys.platform == "win32" else ""
FFMPEG_URLS = ("https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
               "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip")


def tool(ffmpeg, name):
    """ffmpeg/ffprobe: путь из настроек; «auto» — рядом с программой, установленный программой, PATH, C:/ffmpeg."""
    cands = [] if ffmpeg in ("", "auto") else [Path(ffmpeg).with_name(name + EXE)]
    cands += [APP_DIR / "ffmpeg" / f"{name}{EXE}", FFMPEG_HOME / f"{name}{EXE}", shutil.which(name),
              Path("C:/ffmpeg/bin") / f"{name}{EXE}"]
    found = next((str(c) for c in cands if c and Path(c).is_file()), None)
    if not found:
        raise FileNotFoundError(t("{} не найден ({}) — укажите путь к ffmpeg в настройках").format(name, ffmpeg))
    return found


def install_ffmpeg(progress=lambda pct: None):
    """Скачивает сборку ffmpeg и кладёт ffmpeg.exe/ffprobe.exe в FFMPEG_HOME. progress(pct) → True отменяет."""
    if sys.platform != "win32":
        raise RuntimeError(t("Поставьте ffmpeg из репозитория своего дистрибутива, например: sudo apt install ffmpeg"))
    import urllib.request
    import zipfile
    FFMPEG_HOME.mkdir(parents=True, exist_ok=True)
    tmp = FFMPEG_HOME / "download.zip"
    err = None
    for url in FFMPEG_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SmoothCursor"})
            with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
                total, done = int(r.headers.get("Content-Length") or 0), 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress(done * 100 / total if total else 0):
                        raise RuntimeError(t("загрузка отменена"))
            with zipfile.ZipFile(tmp) as z:
                for member in z.namelist():
                    if member.endswith(("/ffmpeg.exe", "/ffprobe.exe")):
                        (FFMPEG_HOME / Path(member).name).write_bytes(z.read(member))
            break
        except RuntimeError:
            raise
        except Exception as e:
            err = e
    tmp.unlink(missing_ok=True)
    if not (FFMPEG_HOME / "ffmpeg.exe").exists():
        raise RuntimeError(t("не удалось скачать ffmpeg: {}").format(err))
    _working.clear()
    return str(FFMPEG_HOME / "ffmpeg.exe")


# Кодировщики по порядку предпочтения: NVIDIA, AMD, Intel, затем процессор (работает везде, но медленнее)
ENCODERS = ("hevc_nvenc", "hevc_amf", "hevc_qsv", "libx264")
_working = {}
_major = {}


def encoder_works(ffmpeg, codec):
    """Пробный кадр: кодировщик может быть в сборке ffmpeg, но без подходящей видеокарты не работать."""
    if (ffmpeg, codec) not in _working:
        r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=s=640x360:d=0.1", "-frames:v", "1",
                            "-c:v", codec, "-f", "null", "-"], capture_output=True, creationflags=CREATE_NO_WINDOW)
        _working[(ffmpeg, codec)] = r.returncode == 0
    return _working[(ffmpeg, codec)]


def ffmpeg_major(ffmpeg):
    """Старшая версия ffmpeg: в дистрибутивах Linux бывает 4 и 6. У ночных сборок вместо номера дата: считаем новыми.
    На Windows ffmpeg ставит сама программа, спрашивать незачем."""
    if sys.platform == "win32":
        return 99
    if ffmpeg not in _major:
        out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW).stdout
        m = re.match(r"ffmpeg version n?(\d+)\.", out)
        _major[ffmpeg] = int(m[1]) if m else 99
    return _major[ffmpeg]


def graph_option(ffmpeg):
    """Как передать граф фильтров файлом: «-/filter_complex» появилась в ffmpeg 7."""
    return "-filter_complex_script" if ffmpeg_major(ffmpeg) < 7 else "-/filter_complex"


def fps_option(ffmpeg):
    """«-fps_mode» появилась в ffmpeg 5.1."""
    return "-vsync" if ffmpeg_major(ffmpeg) < 6 else "-fps_mode"


def pick_encoder(ffmpeg, codec):
    if codec != "auto" and encoder_works(ffmpeg, codec):
        return codec
    return next(c for c in ENCODERS if c == "libx264" or encoder_works(ffmpeg, c))


# Профили ProRes (prores_ks): номер профиля и формат пикселей. CQ и пресет к ним не относятся — качество задаёт профиль
PRORES = {"proxy": (0, "yuv422p10le"), "lt": (1, "yuv422p10le"), "422": (2, "yuv422p10le"),
          "hq": (3, "yuv422p10le"), "4444": (4, "yuv444p10le")}


def out_ext(codec):
    return ".mov" if codec.startswith("prores") else ".mp4"


def disk_check(part, frac, out):
    """Рендер сделан на долю frac: если по нынешнему темпу файл не влезет на диск — стоп сразу, а не через час."""
    size = part.stat().st_size if part.exists() else 0
    need, free = size / max(frac, 1e-6) * 1.05 - size, shutil.disk_usage(part.parent).free
    if need > free:
        old = t(" (старый {} — {:.0f} ГБ, его можно удалить)").format(out.name, out.stat().st_size / 1e9) \
            if out.exists() else ""
        raise RuntimeError(t("не хватит места на диске: рендеру нужно ещё ≈ {:.0f} ГБ, свободно {:.0f} ГБ{}. "
                             "Освободи место или выбери кодек полегче").format(need / 1e9, free / 1e9, old))


def smooth_file(video):
    """Готовый результат записи: самый свежий из _smooth.mp4 и _smooth.mov; нет ни одного — путь .mp4."""
    found = [p for p in (video.with_name(video.stem + "_smooth" + e) for e in (".mp4", ".mov")) if p.exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else video.with_name(video.stem + "_smooth.mp4")


def encoder_args(codec, cq, preset, prores="hq"):
    """Качество одним числом (меньше — лучше) и пресет p1…p7 → параметры конкретного кодировщика."""
    if codec.startswith("prores"):
        prof, pix = PRORES.get(prores, PRORES["hq"])
        return ["-profile:v", str(prof), "-vendor", "apl0", "-pix_fmt", pix]
    p = int(str(preset).lstrip("p") or 5)
    if codec.endswith("_nvenc"):
        return ["-preset", f"p{p}", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    if codec.endswith("_amf"):
        return ["-quality", ("speed", "balanced", "quality")[min(2, (p - 1) // 3)], "-rc", "cqp",
                "-qp_i", str(cq), "-qp_p", str(cq + 2), "-qp_b", str(cq + 4)]
    if codec.endswith("_qsv"):
        return ["-preset", ("veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")[p - 1],
                "-global_quality", str(cq)]
    return ["-preset", ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")[p - 1],
            "-crf", str(cq), "-pix_fmt", "yuv420p"]


def probe(ffprobe, video):
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_packets,pix_fmt,color_space,color_range,"
                        "color_transfer,color_primaries:format=duration",
                        "-of", "json", str(video)], capture_output=True, text=True,
                       creationflags=CREATE_NO_WINDOW)
    if r.returncode:
        raise RuntimeError(t("ffprobe не смог прочитать {}: {}").format(video, r.stderr.strip()))
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"w": s["width"], "h": s["height"], "fps": num / den, "rate": s["r_frame_rate"],
            "frames": int(s["nb_read_packets"]),
            "duration": float(j["format"]["duration"]), "pix_fmt": s.get("pix_fmt", "yuv420p"),
            # без тегов считаем как OBS по умолчанию: BT.709, ограниченный диапазон
            "space": tag(s, "color_space"), "range": "pc" if s.get("color_range") == "pc" else "tv",
            "trc": tag(s, "color_transfer"), "primaries": tag(s, "color_primaries")}


def tag(stream, key):
    v = stream.get(key, "unknown")
    return "bt709" if v in ("unknown", "reserved") else v


def make_sprites(work, types, poses, size, rc, custom=None, morphs=()):
    """PNG для каждого слота; возвращает [(слот, файл, hot_x, hot_y)] в порядке наложения (снизу вверх).
    poses — {(тип, шаг клика, шаг наклона)}, которые есть в видео: спрайты только для них; morphs — {(прежний тип,
    новый тип, шаг смены вида, шаг наклона)}."""
    big = {}

    def big_of(t):  # поворот и масштаб с 4× запасом и уменьшение — без мыла и лесенки
        if t not in big:
            big[t] = wc.cursor_sprite(t, size * 4, custom, rc["cursor_theme"], rc["app_cursor_scale"])
        return big[t]

    def pose(t, lvl, k):
        if not lvl and not k:
            return wc.cursor_sprite(t, size, custom, rc["cursor_theme"], rc["app_cursor_scale"])
        return pose_sprite(*big_of(t), rc, lvl, k)

    mains = {p: pose(*p) for p in sorted({(t, 0, 0) for t in types} | set(poses))}
    slots = []
    if rc["motion_blur"]:  # «призраки» — без анимации клика (при ней их нет), наклонены почти как курсор
        ghosts = {(t, gk): mains.get((t, 0, gk)) or pose(t, 0, gk)
                  for t, gk in sorted({(t, ghost_tilt(k)) for t, lvl, k in mains if not lvl})}
        for gi, a in enumerate(ghost_alphas(rc)):
            for (t, gk), (img, hot) in ghosts.items():
                g = img.copy()
                g.putalpha(img.getchannel("A").point(lambda v: round(v * a)))
                slots.append((f"g{gi}_{t}{tilt_tag(gk)}", g, hot))
    slots += [(f"{t}_{lvl}{tilt_tag(k)}", img, hot) for (t, lvl, k), (img, hot) in mains.items()]
    slots += [(f"m{m}_{a}_{b}{tilt_tag(k)}", *morph_sprite(*big_of(a), *big_of(b), m / MORPH_STEPS, rc["morph_bounce"],
                                                            deg=k * TILT_STEP_DEG)) for a, b, m, k in sorted(morphs)]
    if rc["debug_raw"]:
        img, hot = mains.get(("arrow", 0, 0)) or wc.cursor_sprite("arrow", size, theme=rc["cursor_theme"])
        a = np.array(img, float)
        a[..., :3] = a[..., :3] * 0.3 + np.array([255, 30, 60]) * 0.7  # красный
        a[..., 3] *= 0.6
        slots.append(("raw", Image.fromarray(a.round().astype(np.uint8), "RGBA"), hot))
    out = []
    for name, im, hot in slots:
        im.save(work / f"{name}.png")
        out.append((name, f"{name}.png", *hot))
    return out


def sprite_name(track, n, types, have):
    """Тип курсора в кадре n и его спрайт с учётом шагов анимации клика и наклона (нет такого — обычный курсор)."""
    t = types[track["type"][n]] if track["type"][n] < len(types) else "arrow"
    t = t if f"{t}_0" in have else "arrow"
    if track["morph"][n]:  # смена вида курсора: шаг важнее клика
        m = f"m{track['morph'][n]}_{types[track['prev'][n]]}_{t}{tilt_tag(ghost_tilt(track['tilt'][n]))}"
        if m in have:
            return t, m
    main = f"{t}_{track['level'][n]}{tilt_tag(track['tilt'][n])}"
    return t, main if main in have else f"{t}_0"


BLUR_LAG = 4  # на сколько кадров вперёд ffmpeg может заглянуть, пока ждёт слой точного блюра (с запасом)


def piecewise(vals, n0, fps, t0):
    """Выражение ffmpeg от времени кадра t: значение кадра n0+i — до середины между ним и следующим кадром."""
    e = str(vals[-1])
    for i in range(len(vals) - 2, -1, -1):
        if vals[i] != vals[i + 1]:
            e = f"if(lt(t,{(n0 + i + 0.5) / fps - t0:.6f}),{vals[i]},{e})"
    return e


def write_commands(path, track, info, slots, clip, blur):
    """sendcmd: на каждом кадре двигаем только изменившиеся overlay. clip = (начало, конец) в секундах.
    blur — {кадр: левый верхний угол} слоя с курсором (см. blur_plan); спрайт — только отладочный сырой курсор
    (raw_vx — уже в пикселях видео)."""
    hot = {name: (hx, hy) for name, _, hx, hy in slots if name != "blur"}
    fps, (t0, t1) = info["fps"], clip
    state = {name: None for name in hot}
    pos = lambda k: blur.get(k, (HIDE, 0))
    empty = True
    with open(path, "w", encoding="ascii") as f:
        for n in range(len(track["x"])):
            if not t0 - 1 / fps <= n / fps <= t1 + 1 / fps:
                continue
            want = {}
            if "raw" in hot and 0 <= track["raw_x"][n] < info["dw"] and 0 <= track["raw_y"][n] < info["dh"]:
                want["raw"] = (track["raw_vx"][n], track["raw_vy"][n])
            cmds = []
            for name, cur in state.items():
                new = None
                if name in want:
                    x, y = want[name]
                    new = (round(x - hot[name][0]), round(y - hot[name][1]))
                if new != cur:
                    cmds += [f"overlay@{name} x {new[0]}", f"overlay@{name} y {new[1]}"] if new else [f"overlay@{name} x {HIDE}"]
                    state[name] = new
            if pos(n) != pos(n - 1):
                # Вход слоя — поток, и ffmpeg, прежде чем смешать кадр, заглядывает в следующий: команда кадра n
                # срабатывает до того, как смешаны прошлые. Поэтому позиция — функция времени кадра t с запасом
                # на BLUR_LAG кадров назад: каждый кадр берёт свою, когда бы команда ни сработала.
                for axis in (0, 1):
                    vals = [pos(k)[axis] for k in range(n - BLUR_LAG, n + 1)]
                    cmds.append(f"overlay@blur {'xy'[axis]} '{piecewise(vals, n - BLUR_LAG, fps, t0)}'")
            if cmds:
                # ffmpeg отсчитывает время кадров от начала файла (или от -ss). Окно в 1 с: команда сработает
                # и при выпавших кадрах, а sendcmd не будет перебирать старые интервалы.
                ts = max(0.0, (n - 0.5) / fps - t0)
                f.write(f"{ts:.6f}-{ts + 1:.6f} {', '.join(cmds)};\n")
                empty = False
        if empty:  # курсор всю запись на другом мониторе, а пустой список sendcmd считает ошибкой
            f.write(f"0.000000-1.000000 overlay@blur x {HIDE};\n")


BLUR_CAP, BLUR_SAMPLES = 512, 256  # точный motion blur: слой не больше 512 px, до 256 выборок за выдержку


def blur_plan(track, sprites, types, fps, rc, grid, f0, f1, ghost_ms=()):
    """Курсор — один слой RGBA на кадр, считаем его здесь. Позиция дробная: спрайты через overlay в 4:2:0 встают
    только на чётный пиксель, и на медленном доводе курсор шёл бы ступеньками по 2 px.

    Режимы: точный motion blur, как у камеры (blur_accurate) — курсор усреднён по всем моментам выдержки
    [t − 1/shutter с, t] с шагом ≈ 1 px пути; шлейф копий (ghost_ms — насколько раньше «призраки») — бледные копии
    под курсором; иначе — просто курсор.
    sprites: {имя: (RGBA float32, hotspot)}; grid: (время мс, x, y, виден) траектории с шагом 1 мс в пикселях
    видео; f0…f1 — нужные кадры. Возвращает (ширина, высота слоя, {кадр: левый верхний угол}, байты слоёв).
    Хвост длиннее BLUR_CAP обрезается: при такой скорости он почти прозрачный."""
    tg, gx, gy, gv = grid
    accurate = rc["motion_blur"] and rc["blur_accurate"]
    S, tf = (1000 / rc["shutter"] if accurate else max(ghost_ms, default=0)), track["t"]  # выдержка / длина шлейфа, мс
    n_ghosts = len(ghost_ms)
    prem = {}  # спрайты с премультиплицированной альфой: их можно просто складывать
    for name, (img, hot) in sprites.items():
        a = img[..., 3:] / 255
        prem[name] = (np.concatenate([img[..., :3] * a, a], -1).astype(np.float32), hot)
    # +4: пиксель на субпиксельный сдвиг и пиксель на чётный угол слоя (overlay в 4:2:0 округляет его до чётного)
    sw = max(p.shape[1] for p, _ in prem.values()) + 4
    sh = max(p.shape[0] for p, _ in prem.values()) + 4
    m1 = min(f1, len(tf))  # дальше конца видео — пустые слои (запас, чтобы конец задавало видео)
    # длина пути за выдержку: по ней число выборок и размер слоя (по 99.5% кадров, чтобы один рывок не раздул все)
    probe = tf[f0:m1, None] - S * np.linspace(0, 1, 9)
    px, py = np.interp(probe, tg, gx), np.interp(probe, tg, gy)
    k = np.clip(np.ceil(np.hypot(np.diff(px), np.diff(py)).sum(1)), 1, BLUR_SAMPLES).astype(int) if accurate \
        else np.ones(m1 - f0, int)
    vis = track["visible"][f0:m1]
    pw, ph = sw, sh
    if vis.any():
        pw = int(np.clip(np.percentile(np.ptp(px[vis], 1), 99.5) + sw, sw, BLUR_CAP))
        ph = int(np.clip(np.percentile(np.ptp(py[vis], 1), 99.5) + sh, sh, BLUR_CAP))
    q = lambda v: np.round(v * 16) / 16  # 1/16 px хватает; заодно курсор в покое — ровно на пикселе, без мыла

    def at(ts, name):
        """Левые верхние углы спрайта в моменты ts. Хотспот — целый: в покое (рука на целом пикселе) курсор резкий,
        а не размазан на полпикселя."""
        hx, hy = prem[name][1]
        return q(np.interp(ts, tg, gx) - round(hx)), q(np.interp(ts, tg, gy) - round(hy))

    def stamps(f):
        """Что рисовать в кадре f снизу вверх: [(спрайт, x, y, вес)]; выборки одного спрайта складываются с весами,
        а следующий спрайт ложится поверх. Последний — сам курсор (первая его выборка — нынешний момент)."""
        main = names[f]
        if accurate:
            ts = tf[f] - S * np.arange(k[f - f0]) / k[f - f0]
            ts = ts[gv[np.clip(np.round((ts - tg[0]) / smoothing.STEP_MS).astype(int), 0, len(tg) - 1)]]
            return [(main, *at(ts, main), np.full(len(ts), 1 / k[f - f0]))]
        x, y = at(tf[f:f + 1], main)
        out = []
        # шлейф только в движении и не во время анимации клика: несжатые «призраки» торчали бы из-под наклонённого
        # курсора, а в покое утолщали бы его края
        if n_ghosts and track["level"][f] == 0 and not track["morph"][f]:
            t = sprite_name(track, f, types, prem)[0]
            for gi in range(n_ghosts - 1, -1, -1):  # дальние, бледные — снизу
                g = f"g{n_ghosts - 1 - gi}_{t}{tilt_tag(ghost_tilt(track['tilt'][f]))}"
                gxs, gys = at(tf[f:f + 1] - ghost_ms[gi], g if g in prem else main)
                if g in prem and abs(gxs[0] - x[0]) + abs(gys[0] - y[0]) > 0.5:
                    out.append((g, gxs, gys, np.ones(1)))
        return out + [(main, x, y, np.ones(1))]

    names, origin = {}, {}
    for f in range(f0, m1):
        if track["visible"][f]:
            names[f] = sprite_name(track, f, types, prem)[1]
            st = stamps(f)
            if not st[-1][1].size:  # в момент кадра курсор виден, так что сюда не попадаем — но без падения
                del names[f]
                continue
            h, w = prem[names[f]][0].shape[:2]
            xs = np.floor(np.concatenate([s[1] for s in st]))
            ys = np.floor(np.concatenate([s[2] for s in st]))
            x0, y0 = np.floor(st[-1][1][0]), np.floor(st[-1][2][0])
            # слой целиком вмещает след; если не вмещает — ведущий (нынешний) курсор целый, обрезается хвост.
            # Угол — чётный: overlay в 4:2:0 сам округлил бы нечётный, и курсор прыгал бы на пиксель
            ox = int(min(max(xs.min(), x0 + w + 2 - pw), x0))
            oy = int(min(max(ys.min(), y0 + h + 2 - ph), y0))
            origin[f] = (ox - ox % 2, oy - oy % 2)

    def splat(p, xs, ys, ws):
        """Спрайт в дробных точках xs, ys (относительно слоя) с весами ws: каждую раскладываем на 4 соседних целых
        положения с билинейными весами. Возвращает (рамка x0, y0, x1, y1, сумма) — или None, если вне слоя."""
        h, w = p.shape[:2]
        ix, iy = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - ix, ys - iy
        cx = np.concatenate([ix, ix + 1, ix, ix + 1])
        cy = np.concatenate([iy, iy, iy + 1, iy + 1])
        cw = np.concatenate([(1 - fx) * (1 - fy) * ws, fx * (1 - fy) * ws, (1 - fx) * fy * ws, fx * fy * ws])
        nz = cw > 1e-6
        pos, inv = np.unique(np.stack([cx[nz], cy[nz]], 1), axis=0, return_inverse=True)
        cnt = np.zeros(len(pos))
        np.add.at(cnt, inv.ravel(), cw[nz])
        bx0, by0 = max(pos[:, 0].min(), 0), max(pos[:, 1].min(), 0)
        bx1, by1 = min(pos[:, 0].max() + w, pw), min(pos[:, 1].max() + h, ph)
        if bx0 >= bx1 or by0 >= by1:
            return None
        acc = np.zeros((by1 - by0, bx1 - bx0, 4), np.float32)
        for (x, y), c in zip(pos.tolist(), cnt.tolist()):
            x0, y0, x1, y1 = max(x, bx0), max(y, by0), min(x + w, bx1), min(y + h, by1)
            if x0 < x1 and y0 < y1:
                acc[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] += p[y0 - y:y1 - y, x0 - x:x1 - x] * c
        return bx0, by0, bx1, by1, acc

    def layers():
        empty, still = bytes(pw * ph * 4), {}
        for f in range(f0, f1):
            if f not in names:
                yield empty
                continue
            st = stamps(f)
            ox, oy = origin[f]
            st = [(n, xs - ox, ys - oy, ws) for n, xs, ys, ws in st]
            # курсор стоит — тот же слой, что в прошлый раз на этом месте
            n, xs, ys, ws = st[0]
            key = (n, xs[0], ys[0]) if len(st) == 1 and np.ptp(xs) == np.ptp(ys) == 0 and abs(ws.sum() - 1) < 1e-6 \
                else None
            if key in still:
                yield still[key]
                continue
            out = np.zeros((ph, pw, 4), np.float32)  # премультиплицированный, альфа 0…1
            for n, xs, ys, ws in st:  # следующий спрайт — поверх: out = src + out·(1 − α src)
                s = splat(prem[n][0], xs, ys, ws)
                if s:
                    bx0, by0, bx1, by1, acc = s
                    box = out[by0:by1, bx0:bx1]
                    box *= 1 - acc[..., 3:]
                    box += acc
            a = out[..., 3]
            res = np.zeros((ph, pw, 4), np.uint8)
            res[..., :3] = np.clip(out[..., :3] / np.maximum(a, 1e-6)[..., None] + 0.5, 0, 255)
            res[..., 3] = np.clip(a * 255 + 0.5, 0, 255)  # точный блюр: доля выдержки, когда виден
            buf = res.tobytes()
            if key is not None:
                still[key] = buf
            yield buf

    return pw, ph, origin, layers()


def export_csv(path, track, w, h, fps, types):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "x", "y", "visible", "type", "fusion_x", "fusion_y"])
        for n in range(len(track["x"])):
            x, y = track["vx"][n], track["vy"][n]
            wr.writerow([n, round(n / fps, 5), round(x, 2), round(y, 2), int(track["visible"][n]),
                         types[track["type"][n]], round(x / w, 6), round(1 - y / h, 6)])


def sync_offset(cfg, fps):
    """Сдвиг курсора относительно картинки, мс: по fps видео ({fps: мс, default: мс}) или одно число."""
    offset = cfg["sync"]["offset_ms"]
    return offset.get(round(fps), offset.get("default", 0)) if isinstance(offset, dict) else offset


def cursor_track(lg, info, cfg, ghost_ms=()):
    """Курсор по кадрам видео: позиции в пикселях видео (vx, vy), видимость, сила анимации клика press 0..1 и её шаг
    level, наклон в движении tilt_deg (°) и tilt (шаги TILT_STEP_DEG). Возвращает (track, to_video — дисплей → видео,
    кроп источника в пикселях дисплея, размер курсора на видео в px)."""
    rc, disp, fps = cfg["render"], lg["display"], info["fps"]
    # Где дисплей на холсте OBS (снято с трансформации источника при записи); у старых логов — на весь кадр
    mp = disp.get("map") or {"m": [info["w"] / disp["width"], 0, 0, 0, info["h"] / disp["height"], 0],
                             "canvas": [info["w"], info["h"]], "crop": [0, 0, disp["width"], disp["height"]]}
    a, b, c, d, e, f = mp["m"]
    kx, ky = info["w"] / mp["canvas"][0], info["h"] / mp["canvas"][1]  # холст → видео (масштаб вывода OBS)

    def to_video(x, y):
        return kx * (a * x + b * y + c), ky * (d * x + e * y + f)

    track = smoothing.frame_track(lg, fps, info["frames"], cfg["smoothing"], sync_offset(cfg, fps), ghost_ms)
    press = click_press(track["t"], track["downs"], track["press_lens"], rc["click_ms"]) if rc["click_animation"]         else np.zeros(len(track["x"]))
    track["press"], track["level"] = press, np.round(press * (CLICK_STEPS - 1)).astype(int)
    cl, ct, cr, cb = mp["crop"]  # за кропом источника курсора на видео нет
    track["visible"] &= (track["x"] >= cl) & (track["x"] < cr) & (track["y"] >= ct) & (track["y"] < cb)
    track["vx"], track["vy"] = to_video(track["x"], track["y"])
    track["raw_vx"], track["raw_vy"] = to_video(track["raw_x"], track["raw_y"])
    track["vghosts"] = [to_video(gx, gy) for gx, gy in track["ghosts"]]
    vis = track["visible"]
    v = np.gradient(track["vx"]) * fps / info["w"] if len(vis) > 1 else np.zeros(len(vis))
    # там, где курсор появляется или пропадает, разница позиций — скачок, а не скорость
    ok = vis & np.r_[vis[1:], True] & np.r_[True, vis[:-1]]
    track["type"] = hold_types(track["type"], vis, rc["morph_min_ms"] if rc["morph_hold"] else 0, fps)
    track["morph"], track["prev"] = morph_steps(track["type"], vis, fps, rc["morph_ms"]) if rc["morph_animation"]         else (np.zeros(len(vis), int), track["type"])
    # смена вида курсора наклон не трогает: спрайты перехода наклоняются как обычные
    track["tilt_deg"] = np.where(vis, frame_tilt(v, ok, fps, rc["motion_tilt_right_deg"], rc["motion_tilt_left_deg"],
                                                 step=None), 0.0)
    track["tilt"] = click_tilt(np.round(track["tilt_deg"] / TILT_STEP_DEG).astype(int), track["level"])
    zoom = abs(kx * ky * (a * e - b * d)) ** 0.5  # во сколько раз захват увеличен на видео
    size = max(4, round(wc.cursor_base_size() * disp["dpi"] / 96 * rc["cursor_scale"] * zoom))
    return track, to_video, (cl, ct, cr, cb), size


def eta_left(start_pct, pct, elapsed):
    """Сколько секунд осталось при той же скорости; None — ещё рано судить."""
    if pct - start_pct < 1 or elapsed < 1:
        return None
    return elapsed / (pct - start_pct) * max(100 - pct, 0)


def eta_text(seconds):
    """Примерное время: чем дальше конец, тем грубее — иначе цифры прыгают."""
    if seconds is None:
        return t("считаю время…")
    if seconds < 5:
        return t("почти готово")
    if seconds < 60:
        return t("осталось ≈ {} с").format(max(5, round(seconds / 5) * 5))
    if seconds < 600:
        m, s = divmod(round(seconds / 10) * 10, 60)
        return t("осталось ≈ {} мин {} с").format(m, s) if s else t("осталось ≈ {} мин").format(m)
    if round(seconds / 60) < 60:
        return t("осталось ≈ {} мин").format(round(seconds / 60))
    h, m = divmod(round(seconds / 300) * 5, 60)  # больше часа — с точностью до 5 мин
    return t("осталось ≈ {} ч {} мин").format(h, m) if m else t("осталось ≈ {} ч").format(h)


def blank_frames(ffmpeg, result):
    """Есть ли в результате обнулённые кадры (Y≈0 — зелёный экран). В ограниченном диапазоне (tv, как пишет OBS) даже
    чёрный — это Y=16, так что средняя яркость ниже 8 у настоящей картинки не бывает. Читаем только ключевые кадры:
    один быстрый проход без исходника.
    ponytail: сбой, начавшийся после последнего ключевого кадра (обычно последние ~10 с), не заметим."""
    r = subprocess.run([ffmpeg, "-v", "error", "-discard", "nokey", "-skip_frame", "nokey", "-i", str(result), "-vf",
                        "scale=64:36,format=yuv420p", fps_option(ffmpeg), "passthrough", "-f", "rawvideo", "-"],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    frames = np.frombuffer(r.stdout, np.uint8)
    frames = frames[:len(frames) // 3456 * 3456].reshape(-1, 3456)[:, :2304]  # кадр 64×36 yuv420p; Y — первые 2304 байта
    return bool(len(frames)) and bool((frames.mean(axis=1) < 8).any())


def render(video, log_path, cfg, progress=None, clip=None, hwaccel=True):
    """Рендер видео с плавным курсором. progress(pct) → True отменяет рендер (по умолчанию — лог каждые 10%).
    clip = (начало_с, длина_с) — короткое превью во временный файл. hwaccel=False — декод исходника процессором."""
    if progress is None:
        nxt = [0]

        def progress(pct):
            if pct >= nxt[0]:
                log.info(t("Рендер %d%%"), min(pct, 100))
                nxt[0] = pct // 10 * 10 + 10

    rc = cfg["render"]
    ffmpeg, ffprobe = tool(rc["ffmpeg"], "ffmpeg"), tool(rc["ffmpeg"], "ffprobe")
    codec = pick_encoder(ffmpeg, rc["codec"])
    if codec == "prores_ks" and rc["prores_fast"] and encoder_works(ffmpeg, "prores_aw"):
        codec = "prores_aw"  # тот же ProRes, кодирует в разы быстрее; prores_ks почти всё время рендера и съедал
    info = probe(ffprobe, video)
    lg = json.loads(Path(log_path).read_text(encoding="utf-8"))
    info["dw"], info["dh"] = lg["display"]["width"], lg["display"]["height"]
    fps = info["fps"]
    offset = sync_offset(cfg, fps)
    if clip:
        start = min(max(0.0, clip[0]), max(0.0, info["duration"] - 0.5))
        clip = (start, min(clip[1], info["duration"] - start))
    span = (clip[0], clip[0] + clip[1]) if clip else (0.0, float("inf"))
    duration = clip[1] if clip else info["duration"]
    log.info(t("%s %s: %dx%d @ %.3g fps, %s, offset %g мс, кодек %s"), t("Превью" if clip else "Рендер"), video.name,
             info["w"], info["h"], fps, t("{}–{} с").format(f"{clip[0]:.1f}", f"{span[1]:.1f}") if clip
             else t("{} кадров").format(info["frames"]), offset, codec)

    accurate = rc["motion_blur"] and rc["blur_accurate"]  # точный motion blur вместо шлейфа из копий
    ghost_ms = [1000 / fps * rc["blur_length"] * (i + 1) / len(ghost_alphas(rc))
                for i in range(len(ghost_alphas(rc)))] if rc["motion_blur"] and not accurate else []
    track, to_video, (cl, ct, cr, cb), size = cursor_track(lg, info, cfg, ghost_ms)
    vis, types = track["visible"], lg["types"]
    used = sorted({types[k] for k in np.unique(track["type"][vis])} | {"arrow"}, key=types.index)
    mv = vis & (track["morph"] > 0)
    poses = {(types[k], lvl, tl)
             for k, lvl, tl in zip(*(track[c][vis & ~mv].tolist() for c in ("type", "level", "tilt")))}
    morphs = {(types[a], types[b], m, ghost_tilt(k))  # наклон — шагом шлейфа: слоёв в ffmpeg в разы меньше
              for a, b, m, k in zip(*(track[c][mv].tolist() for c in ("prev", "type", "morph", "tilt")))}
    if clip:
        out = Path(tempfile.gettempdir()) / f"{video.stem}_preview{out_ext(codec)}"
        try:
            out.unlink(missing_ok=True)
        except OSError:  # прошлое превью ещё открыто в плеере
            out = out.with_name(f"{video.stem}_preview_{time.strftime('%H%M%S')}{out_ext(codec)}")
    else:
        out = video.with_name(video.stem + ("_smooth_debug" if rc["debug_raw"] else "_smooth") + out_ext(codec))
        if rc["export_keyframes"]:
            export_csv(video.with_name(video.stem + "_cursor.csv"), track, info["w"], info["h"], fps, types)

    with tempfile.TemporaryDirectory(prefix="smooth_cursor_") as tmp:
        work = Path(tmp)
        slots = make_sprites(work, used, poses, size, {**rc, "motion_blur": rc["motion_blur"] and not accurate},
                             lg.get("custom"), morphs)
        # курсор — один слой, кадры которого считаем здесь и подаём ffmpeg через stdin: так он встаёт куда надо
        # с точностью до долей пикселя, а спрайты через overlay в 4:2:0 — только на чётный пиксель
        sprites = {name: (np.asarray(Image.open(work / png), np.float32), (hx, hy))
                   for name, png, hx, hy in slots if name != "raw"}
        slots = [("blur", None, 0, 0)] + [s for s in slots if s[0] == "raw"]
        f0 = int(np.ceil(span[0] * fps - 1e-6)) if clip else 0  # первый кадр превью
        f1 = (int(span[1] * fps) + 3 if clip else info["frames"]) + 2  # слоёв — с запасом: конец задаёт видео
        tg, sx, sy, gvis = track["path"]
        gv = gvis & (sx >= cl) & (sx < cr) & (sy >= ct) & (sy < cb)
        pw, ph, blur, layers = blur_plan(track, sprites, types, fps, rc, (tg, *to_video(sx, sy), gv), f0, f1,
                                         ghost_ms)
        if accurate:
            log.info(t("Точный motion blur: выдержка 1/%d с, слой %dx%d px"), rc["shutter"], pw, ph)
        blur_in = ["-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{pw}x{ph}", "-framerate",
                   info["rate"], "-i", "pipe:0"]
        blur_off = (f0 - 0.5) / fps - span[0]
        write_commands(work / "cmds.txt", track, info, slots, span, blur)
        # Накладываем прямо в YUV исходника: через RGB (format=auto) цвета видео чуть съезжают
        pf = info["pix_fmt"]
        sub = "444" if "444" in pf else "422" if "422" in pf else "420"
        deep = "10" in pf or "12" in pf or "16" in pf
        # pc: ffmpeg до 7 декодирует полный диапазон как yuvj, и format=yuv…p молча сжал бы его в ограниченный (цвет
        # бледнеет, а метка остаётся pc). Явный pc → pc — пересчёта нет; в новом ffmpeg это просто ничего не делает
        keep = "scale=in_range=pc:out_range=pc," if info["range"] == "pc" else ""
        chain = [f"[0:v]{keep}format=yuv{sub}p{'10le' if deep else ''},sendcmd=f=cmds.txt[v0]"]
        for i, (name, *_) in enumerate(slots, 1):
            inp = f"[{i}:v]"
            if name == "blur":
                # Слой кадра n — на полкадра раньше самого кадра (как окна sendcmd), чтобы кадр видео взял именно
                # его. Сдвиг — в микросекундах: в шкале самого слоя (1/fps) он округлился бы до целого кадра.
                chain.append(f"{inp}settb=AVTB,setpts=PTS+({blur_off:.6f})/TB[bl]")
                inp = "[bl]"
            chain.append(f"[v{i - 1}]{inp}overlay@{name}=x={HIDE}:y=0:format=yuv{sub}{'p10' if deep else ''}"
                         f":eof_action=repeat{':shortest=1' if name == 'blur' else ''}[v{i}]")  # видео кончилось — всё
        last, rng = f"[v{len(slots)}]", info["range"]
        if codec.startswith("prores") and rng == "pc":  # ProRes — только ограниченный диапазон: пересчитываем
            chain.append(f"{last}scale=in_range=pc:out_range=tv[vtv]")
            last, rng = "[vtv]", "tv"
        (work / "graph.txt").write_text(";".join(chain), encoding="ascii")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
        if hwaccel:
            cmd += ["-hwaccel", "auto"]
        if clip:
            cmd += ["-ss", f"{clip[0]:.3f}", "-t", f"{clip[1]:.3f}"]
        cmd += ["-i", str(video)]
        for name, png, *_ in slots:
            cmd += blur_in if name == "blur" else ["-i", png]
        cmd += [graph_option(ffmpeg), "graph.txt", "-map", last, "-map", "0:a?", "-c:a", "copy",
                "-c:v", codec, *encoder_args(codec, rc["cq"], rc["preset"], rc["prores"]), fps_option(ffmpeg), "passthrough",
                "-colorspace", info["space"], "-color_range", rng,
                "-color_trc", info["trc"], "-color_primaries", info["primaries"]]
        if codec.startswith("hevc"):
            cmd += ["-tag:v", "hvc1"]
        part = out.with_name(out.stem + ".part" + out.suffix)  # оборванный рендер не затрёт готовый и не выдаст себя за него
        cmd.append(str(part))
        with open(work / "ffmpeg.log", "w+", encoding="utf-8", errors="replace") as err:
            p = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=err, text=True,
                                 stdin=subprocess.PIPE if layers else None, creationflags=CREATE_NO_WINDOW)
            failed = []
            if layers:
                def feed():  # в отдельном потоке: ffmpeg берёт слои по мере надобности, прогресс читаем здесь
                    try:
                        for buf in layers:
                            p.stdin.buffer.write(buf)
                        p.stdin.close()
                    except OSError:  # ffmpeg уже закрылся (отмена, ошибка) — причину скажет он сам
                        pass
                    except Exception as e:  # сбой расчёта слоя — нельзя молча доделать видео без курсора
                        failed.append(e)
                        p.kill()

                threading.Thread(target=feed, daemon=True).start()
            try:
                next_check = 0.02
                for line in p.stdout:
                    if line.startswith("out_time_us=") and line[12:].strip().isdigit():
                        pct = int(line[12:]) / 1e4 / duration
                        if progress(pct):
                            raise RuntimeError(t("рендер отменён"))
                        if pct / 100 >= next_check:  # хватит ли места — видно по первым процентам, а не через час
                            next_check += 0.05
                            disk_check(part, pct / 100, out)
                if failed:
                    raise failed[0]
                if p.wait():
                    err.seek(0)
                    tail = "".join(err.readlines()[-15:])
                    raise RuntimeError(t("ffmpeg завершился с кодом {}:\n{}").format(p.returncode, tail))
            except BaseException:
                p.kill()
                p.wait()
                part.unlink(missing_ok=True)
                raise
    if hwaccel and info["range"] == "tv" and blank_frames(ffmpeg, part):
        # Сбой драйвера видеокарты (у NVIDIA — nvlddmkm Xid) молча обнуляет кадры аппаратного декодера: ffmpeg
        # завершается без ошибки, а на выходе зелёный экран с одним курсором. Переделываем с декодом процессором.
        part.unlink(missing_ok=True)
        log.warning(t("Видеокарта отдала пустые кадры — рендерю заново без аппаратного декодирования"))
        return render(video, log_path, cfg, progress, clip, hwaccel=False)
    try:
        os.replace(part, out)
    except OSError:  # старый результат открыт в плеере — новый не выбрасываем
        raise RuntimeError(t("{} открыт в другой программе — новый рендер сохранён как {}").format(out.name, part.name))
    progress(100)

    o = probe(ffprobe, out)
    log.info(t("Готово: %s — %dx%d @ %.3g fps, %.2f с"), out, o["w"], o["h"], o["fps"], o["duration"])
    if (o["w"], o["h"]) != (info["w"], info["h"]) or abs(o["duration"] - duration) > 1.5 / fps:
        log.warning(t("Параметры результата не совпадают с исходником (%.2f с)!"), duration)
    return out
