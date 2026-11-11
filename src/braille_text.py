"""Turn detected Braille cells (boxes + dot classes) into text.

1. Group cells into lines by vertical position, sort each line left -> right.
2. Insert a space where neighbouring cell centres are > 1.5 character pitches apart.
3. Map dot patterns to characters with the grade-1 tables of the Angelina dataset
   (Ovodov, src/letters.py): Russian or English alphabet, punctuation, the number sign
   (dots 3-4-5-6: following a-j are digits), the Russian capital sign (dots 4-5) and the
   English capital sign (dot 6; doubled = whole word).
Unknown patterns are shown as the Unicode Braille character itself.
"""
import numpy as np

SYM = {"256": ".", "2": ",", "25": ":", "26": "?", "23": ";", "235": "!", "2356": "()", "126": "(",
       "345": ")", "36": "-", "34": "/"}
RU = {"1": "а", "12": "б", "2456": "в", "1245": "г", "145": "д", "15": "е", "16": "ё", "245": "ж",
      "1356": "з", "24": "и", "12346": "й", "13": "к", "123": "л", "134": "м", "1345": "н", "135": "о",
      "1234": "п", "1235": "р", "234": "с", "2345": "т", "136": "у", "124": "ф", "125": "х", "14": "ц",
      "12345": "ч", "156": "ш", "1346": "щ", "12356": "ъ", "2346": "ы", "23456": "ь", "246": "э",
      "1256": "ю", "1246": "я", "236": "«", "356": "»", "4": "'", "456": "|", "346": "§"}
EN = {"1": "a", "12": "b", "14": "c", "145": "d", "15": "e", "124": "f", "1245": "g", "125": "h",
      "24": "i", "245": "j", "13": "k", "123": "l", "134": "m", "1345": "n", "135": "o", "1234": "p",
      "12345": "q", "1235": "r", "234": "s", "2345": "t", "136": "u", "1236": "v", "2456": "w",
      "1346": "x", "13456": "y", "1356": "z", "3": "'", "236": "«", "356": "»"}
NUM = {"1": "1", "12": "2", "14": "3", "145": "4", "15": "5", "124": "6", "1245": "7", "125": "8",
       "24": "9", "245": "0"}
NUM_SIGN, CAPS_RU, CAPS_EN = "3456", "45", "6"  # EN capital sign: dot 6 (double = word)


def dots(cls):
    v = int(cls) + 1
    return "".join(str(i + 1) for i in range(6) if v >> i & 1)


def unicode_cell(cls):
    return chr(0x2800 + int(cls) + 1)


def group_lines(boxes):
    """Return list of lines, each a list of indices sorted left -> right."""
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = np.median(boxes[:, 3] - boxes[:, 1])
    order = np.argsort(cy)
    lines, cur, cur_y = [], [order[0]], cy[order[0]]
    for i in order[1:]:
        if cy[i] - cur_y > 0.6 * h:
            lines.append(cur)
            cur = [i]
        else:
            cur.append(i)
        cur_y = np.mean(cy[cur])
    lines.append(cur)
    return [sorted(l, key=lambda i: boxes[i, 0]) for l in lines]


def decode_line(boxes, classes, idx, lang, pitch):
    alpha = RU if lang == "ru" else EN
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    out, num_mode, caps, caps_word = [], False, False, False
    prev = None
    for i in idx:
        # a space is an empty cell: centre distance ~2 pitches instead of 1
        if prev is not None and cx[i] - cx[prev] > 1.5 * pitch:
            out.append(" ")
            num_mode = caps_word = False
        prev = i
        d = dots(classes[i])
        if d == NUM_SIGN:
            num_mode = True
            continue
        if (lang == "ru" and d == CAPS_RU) or (lang == "en" and d == CAPS_EN):
            if caps and lang == "en":  # two capital signs: whole word in capitals
                caps_word = True
            caps = True
            continue
        if caps_word:
            caps = True
        if num_mode and d in NUM:
            out.append(NUM[d])
            continue
        num_mode = False
        ch = alpha.get(d) or SYM.get(d) or unicode_cell(classes[i])
        if caps:
            ch, caps = ch.upper(), False
        out.append(ch)
    return "".join(out)


def page_text(boxes, classes, lang="ru"):
    boxes, classes = np.asarray(boxes), np.asarray(classes)
    lines = group_lines(boxes)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 if len(boxes) else np.zeros(0)
    d = np.concatenate([np.diff(cx[l]) for l in lines if len(l) > 1] or [np.zeros(0)])
    d = d[d > 0]
    # character pitch: typical centre distance of neighbouring cells inside a word
    pitch = np.percentile(d, 30) if len(d) else 1.0
    return "\n".join(decode_line(boxes, classes, l, lang, pitch) for l in lines)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def page_braille(boxes, classes):
    """Same line/space layout as page_text, but as Unicode Braille characters."""
    boxes, classes = np.asarray(boxes), np.asarray(classes)
    lines = group_lines(boxes)
    if not lines:
        return ""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    d = np.concatenate([np.diff(cx[l]) for l in lines if len(l) > 1] or [np.zeros(0)])
    d = d[d > 0]
    pitch = np.percentile(d, 30) if len(d) else 1.0
    out = []
    for l in lines:
        s, prev = [], None
        for i in l:
            if prev is not None and cx[i] - cx[prev] > 1.5 * pitch:
                s.append(" ")
            s.append(unicode_cell(classes[i]))
            prev = i
        out.append("".join(s))
    return "\n".join(out)
