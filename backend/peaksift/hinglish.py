"""Devanagari → Hinglish: write spoken Hindi the way people type it ("kya scene hai bhai").

Whisper transcribes Hindi in Devanagari. Short-form audiences read Hinglish
(Roman script), and the caption fonts don't carry Devanagari glyphs anyway, so
Hindi transcripts are converted word by word. Converting per word keeps every
word's timing, which the captions, clip boundaries and highlight signals rely
on.

The rules aim at everyday Hinglish, not academic transliteration:

* the inherent "a" is dropped where Hindi speakers drop it (schwa deletion):
  करना → karna, समझना → samajhna, अपना → apna, but रहा → raha;
* long vowels are doubled inside a word (आज → aaj, पीना → peena) and shortened
  at its end (करना → karna, नहीं → nahin, की → ki);
* nukta letters become their usual Latin forms (ज़ → z, फ़ → f, ड़ → d);
* English words Whisper spells in Devanagari (गेम, प्लेयर, सब्सक्राइब) come back
  as English through a small loanword list.
"""

from __future__ import annotations

import re
import unicodedata

_INDEPENDENT_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ऑ": "o",
    "ऍ": "e",
    "ऎ": "e",
    "ऒ": "o",
}
_MATRAS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "oo",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॉ": "o",
    "ॅ": "e",
    "ॆ": "e",
    "ॊ": "o",
}
_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "n",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "n",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
}
#: Consonant + nukta.
_NUKTA_FORMS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "ज": "z",
    "फ": "f",
    "ड": "d",
    "ढ": "dh",
    "य": "y",
    "र": "r",
}
_PRECOMPOSED_NUKTA = {
    "क़": "क",
    "ख़": "ख",
    "ग़": "ग",
    "ज़": "ज",
    "ड़": "ड",
    "ढ़": "ढ",
    "फ़": "फ",
    "य़": "य",
}
_NUKTA = "़"
_VIRAMA = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_VISARGA = "ः"
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: English words Whisper writes in Devanagari, mostly from gaming/streaming talk.
LOANWORDS = {
    "गेम": "game",
    "गेमिंग": "gaming",
    "प्लेयर": "player",
    "वीडियो": "video",
    "विडियो": "video",
    "चैनल": "channel",
    "सब्सक्राइब": "subscribe",
    "लाइक": "like",
    "कमेंट": "comment",
    "शेयर": "share",
    "ओके": "ok",
    "ओक": "ok",
    "सॉरी": "sorry",
    "थैंक": "thank",
    "यू": "you",
    "ब्रो": "bro",
    "गाइज": "guys",
    "गाइज़": "guys",
    "लेवल": "level",
    "बॉस": "boss",
    "किल": "kill",
    "मैप": "map",
    "टीम": "team",
    "एनिमी": "enemy",
    "गन": "gun",
    "शॉट": "shot",
    "हेडशॉट": "headshot",
    "मैच": "match",
    "विन": "win",
    "लाइव": "live",
    "स्ट्रीम": "stream",
    "चैट": "chat",
    "फाइनल": "final",
    "राउंड": "round",
    "अपडेट": "update",
    "मिशन": "mission",
    "कार": "car",
    "पुलिस": "police",
    "फोन": "phone",
    "फ़ोन": "phone",
    "मोबाइल": "mobile",
    "प्लान": "plan",
    "सीन": "scene",
    "प्रॉब्लम": "problem",
    "प्लीज": "please",
    "प्लीज़": "please",
    "वेट": "wait",
    "हेलो": "hello",
    "हैलो": "hello",
    "बाय": "bye",
    "सर्वर": "server",
    "लैग": "lag",
    "पिंग": "ping",
    "रैंक": "rank",
    "स्किन": "skin",
    "क्लिप": "clip",
    "डोनेशन": "donation",
    "सुपरचैट": "superchat",
    "मॉडरेटर": "moderator",
    "ऑनलाइन": "online",
    "ऑफलाइन": "offline",
    "इंटरनेट": "internet",
    "ट्रिक": "trick",
    "प्रो": "pro",
    "नूब": "noob",
    "हैक": "hack",
    "हैकर": "hacker",
    "बग": "bug",
    "ग्लिच": "glitch",
    "रिवॉर्ड": "reward",
    "बोनस": "bonus",
    "ऑप": "op",
    "जीटीए": "GTA",
    "माइनक्राफ्ट": "Minecraft",
    "फ्री": "free",
    "फायर": "fire",
    "बैटल": "battle",
    "ग्राउंड": "ground",
    "स्नाइपर": "sniper",
    "राइफल": "rifle",
    "ग्रेनेड": "grenade",
    "ज़ोन": "zone",
    "जोन": "zone",
    "लूट": "loot",
}


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def _tokens(word: str) -> list[list]:
    """Split a Devanagari word into [kind, latin, inherent] tokens.

    kind: "C" consonant, "V" vowel, "N" nasal, "H" visarga, "X" anything else.
    """
    word = unicodedata.normalize("NFC", word)
    for composed, base in _PRECOMPOSED_NUKTA.items():
        word = word.replace(composed, base + _NUKTA)
    out: list[list] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _CONSONANTS:
            latin = _CONSONANTS[ch]
            j = i + 1
            if j < len(word) and word[j] == _NUKTA:
                latin = _NUKTA_FORMS.get(ch, latin)
                j += 1
            out.append(["C", latin, False])
            if j < len(word) and word[j] in _MATRAS:
                out.append(["V", _MATRAS[word[j]], False])
                j += 1
            elif j < len(word) and word[j] == _VIRAMA:
                j += 1
            else:
                out.append(["V", "a", True])
            i = j
            continue
        if ch in _INDEPENDENT_VOWELS:
            out.append(["V", _INDEPENDENT_VOWELS[ch], False])
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            out.append(["N", "n", False])
        elif ch == _VISARGA:
            out.append(["H", "h", False])
        elif ch == _NUKTA:
            pass
        else:
            out.append(["X", ch.translate(_DIGITS), False])
        i += 1
    return out


def _delete_schwas(tokens: list[list]) -> None:
    """Hindi schwa deletion, right to left: drop an inherent 'a' in V C _ C V."""
    vowel_positions = [i for i, t in enumerate(tokens) if t[0] == "V"]
    if not vowel_positions:
        return
    # Word-final inherent schwa goes, unless it's the word's only vowel (न → na).
    last = vowel_positions[-1]
    if (
        tokens[last][2]
        and len(vowel_positions) > 1
        and all(t[0] != "V" for t in tokens[last + 1 :])
    ):
        tokens[last][1] = ""
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        if token[0] != "V" or not token[2] or not token[1]:
            continue
        if i < 2 or tokens[i - 1][0] != "C":
            continue
        before = tokens[i - 2]
        if not (before[0] == "V" and before[1]) and before[0] != "N":
            continue
        if i + 2 >= len(tokens) or tokens[i + 1][0] != "C":
            continue
        after = tokens[i + 2]
        if after[0] == "V" and after[1]:
            token[1] = ""


def _render(tokens: list[list]) -> str:
    # Index of the last vowel actually pronounced, for end-of-word shortening.
    spoken = [i for i, t in enumerate(tokens) if t[0] == "V" and t[1]]
    last = spoken[-1] if spoken else -1
    parts: list[str] = []
    for i, (kind, latin, _inherent) in enumerate(tokens):
        if kind == "V" and latin:
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            at_end = i == last and all(t[0] in ("N", "H", "X") for t in tokens[i + 1 :])
            if at_end:
                latin = {"aa": "a", "ee": "i", "oo": "u" if not next_token else "oo"}.get(
                    latin, latin
                )
            elif latin == "aa" and next_token is not None and next_token[0] == "V":
                latin = "a"  # भाई → bhai, not bhaai
        if kind == "C" and latin == "v" and i == 0:
            nxt = tokens[1][1] if len(tokens) > 1 else ""
            if nxt in ("o", "aa", "a", "ah"):
                latin = "w"  # वो → wo, वाला → wala
        before_labial = (
            i + 1 < len(tokens)
            and tokens[i + 1][0] == "C"
            and tokens[i + 1][1][:1] in ("p", "b", "m")
        )
        if kind == "N" and before_labial:
            latin = "m"  # अंबर → ambar
        parts.append(latin)
    return "".join(parts)


def transliterate_word(word: str) -> str:
    """One word (punctuation attached is kept) from Devanagari to Hinglish."""
    if not contains_devanagari(word):
        return word
    match = re.match(r"^(\W*)(.*?)(\W*)$", word, re.UNICODE)
    lead, core, trail = (
        (match.group(1), match.group(2), match.group(3)) if match else ("", word, "")
    )
    # \W treats Devanagari combining marks as non-word characters; give any that
    # were stripped off the end back to the core.
    while trail and unicodedata.category(trail[0]).startswith("M"):
        core, trail = core + trail[0], trail[1:]
    trail = trail.replace("।", ".").replace("॥", ".")
    key = unicodedata.normalize("NFC", core)
    if key in LOANWORDS:
        return lead + LOANWORDS[key] + trail
    tokens = _tokens(core)
    _delete_schwas(tokens)
    return lead + _render(tokens) + trail


def transliterate_text(text: str) -> str:
    return " ".join(transliterate_word(w) for w in text.split())
