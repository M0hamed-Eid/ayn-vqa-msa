import re
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Token patterns (True / False)
# -----------------------------
TRUE_TOKENS = [
    r"\btrue\b",
    r"\byes\b",
    r"\bصح\b",
    r"\bصحيح\b",
    r"\bصحيحة\b",
    r"\bالصحيح\b",
    r"\bالخطأ\b",
]

FALSE_TOKENS = [
    r"\bfalse\b",
    r"\bfalsy\b",
    r"\bno\b",
    r"\bخطأ\b",
     r"\bالخطأ\b",
    r"\bغلط\b",
    r"\bغير\s+صحيح(?:ة)?\b",
    r"\bغير\s+صحيحة\b",
    r"\bخاطئ\b",
    r"\bخاطئة\b",
    
]


# ---------------------------------------
# Abstain / uncertain (cannot determine)
# ---------------------------------------
ABSTAIN_PATTERNS = [
    # English
    r"\b(can(?:not|'t)\s+determine|can(?:not|'t)\s+tell|not\s+enough\s+information|cannot\s+be\s+sure|unclear)\b",
    # Arabic
    r"لا\s+يمكن(?:نا)?\s+الجزم",
    r"لا\s+يمكن\s+الجزم",
    r"لا\s+يمكن\s+تحديد.*(?:صحة|خطأ|صحيح|خاطئ|العبارة)",
    r"لا\s+نستطيع\s+التأكد",
    r"لا\s+يمكن\s+الحكم",
    #r"غير\s+واضح",
]


# ---------------------------------------
# Strong cues that often precede the label
# ---------------------------------------
STRONG_CUES = [
    # English
    r"therefore[,:\s]*",
    r"final\s+answer[,:\s]*",
    r"the\s+answer\s+is[,:\s]*",
    r"correct\s+answer\s+is[,:\s]*",
    r"so\s+the\s+answer\s+is[,:\s]*",
    r"conclusion[,:\s]*",
    r"verdict[,:\s]*",
    r"determination[,:\s]*",
    r"final[,:\s]*(?:answer)?[,:\s]*",

    # NEW
    r"the\s+statement\s+is\s*[:\-–—,]?\s*",

    # Arabic
    r"الإجابة\s+الصحيحة\s*(?:هي)?\s*[:：]?\s*",
    r"الجواب\s+الصحيح\s*(?:هو|هي)?\s*[:：]?\s*",
    r"الإجابة\s*(?:هي)?\s*[:：]?\s*",
    r"الجواب\s*(?:هو|هي)?\s*[:：]?\s*",
    r"إذًا\s*(?:الجواب|الجواب\s+هو|الإجابة|الإجابة\s+هي)?\s*[:：]?\s*",
    r"الإجابة\s+النهائية\s*(?:هي)?\s*[:：]?\s*"
]


# -----------------------------
# Result type
# -----------------------------
@dataclass
class EvalResult:
    pred: Optional[str]   # "true" / "false" / None
    confidence: float     # 0..1
    needs_review: bool
    reason: str
    conflict: bool 


# -----------------------------
# Helpers
# -----------------------------
def _normalize(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("\u200f", "").replace("\u200e", "")  # RTL marks
    t = t.lower()
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _strip_code_and_quotes(text: str) -> str:
    t = text or ""
    # Remove fenced code blocks
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    # Remove quoted substrings (often repeats the statement)
    t = re.sub(r"\".*?\"", " ", t, flags=re.DOTALL)
    t = re.sub(r"“.*?”", " ", t, flags=re.DOTALL)
    return t


def _match_label(fragment: str) -> Optional[str]:
    for pat in TRUE_TOKENS:
        if re.search(pat, fragment, flags=re.IGNORECASE):
            return "true"
    for pat in FALSE_TOKENS:
        if re.search(pat, fragment, flags=re.IGNORECASE):
            return "false"
    return None


def _has_any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# -----------------------------
# Main evaluator
# -----------------------------
# def evaluate_tf(response: str) -> EvalResult:
#     raw = response or ""
#     if not raw.strip():
#         return EvalResult(None, 0.0, True, "empty_response", conflict=False)

#     text_noquotes = _normalize(_strip_code_and_quotes(raw))

#     # 0) Abstain / uncertain
#     for pat in ABSTAIN_PATTERNS:
#         if re.search(pat, text_noquotes, flags=re.IGNORECASE):
#             return EvalResult(None, 0.0, True, "abstain_or_uncertain", conflict=False)
    
#     best = None  # (score, label, reason)
    
#     m = re.match(
#     r"^\s*[\*\s_`]*((?:true|false)|(?:صح|صحيح|صحيحة)|(?:خطأ|غلط|خاطئ|خاطئة))\b",
#     text_noquotes,
#     flags=re.IGNORECASE,
# )
#     if m: 
#         head = m.group(1)
#         label = _match_label(head)
#         if label: 
#             cand = (3.0, label, "leading_label")
#             best = cand
    

#     # 1) Strong cue windows (highest precision)
#     for cue in STRONG_CUES:
#         for m in re.finditer(cue, text_noquotes, flags=re.IGNORECASE):
#             start = m.end()
#             window = text_noquotes[start:start + 140]
#             label = _match_label(window)
#             if label:
#                 cand = (3.0, label, f"strong_cue:{cue}")
#                 best = max(best, cand, key=lambda x: x[0]) if best else cand

#     # 2) Standalone label lines near end (medium precision)
#     lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
#     tail_lines = lines[-25:]

#     standalone_patterns = [
#     # English
#     (re.compile(r"^[\*\s_`]*true\s*[\.!\?]*[\*\s_`]*$", re.IGNORECASE), "true"),
#     (re.compile(r"^[\*\s_`]*false\s*[\.!\?]*[\*\s_`]*$", re.IGNORECASE), "false"),

#     # Arabic
#     (re.compile(r"^[\*\s_`]*صح\s*[\.!\?]*[\*\s_`]*$"), "true"),
#     (re.compile(r"^[\*\s_`]*(خطأ|غلط)\s*[\.!\?]*[\*\s_`]*$"), "false"),
#     (re.compile(r"^[\*\s_`]*(خطأ|غلط|خاطئ|خاطئة)\s*[\.!\?]*[\*\s_`]*$"), "false")
# ]

#     if not best or best[0] < 3.0:
#         for ln in reversed(tail_lines):
#             ln_norm = _normalize(ln)
#             for rgx, lab in standalone_patterns:
#                 if rgx.match(ln_norm):
#                     cand = (2.0, lab, "standalone_label_line")
#                     best = max(best, cand, key=lambda x: x[0]) if best else cand
#                     break
#             if best and best[0] >= 2.0:
#                 break

#     # 3) Fallback: last occurrence in last N chars (lowest precision)
#     if not best:
#         tail = text_noquotes[-450:]
#         matches = []
#         for pat in TRUE_TOKENS:
#             for mm in re.finditer(pat, tail, flags=re.IGNORECASE):
#                 matches.append((mm.start(), "true"))
#         for pat in FALSE_TOKENS:
#             for mm in re.finditer(pat, tail, flags=re.IGNORECASE):
#                 matches.append((mm.start(), "false"))

#         if matches:
#             matches.sort(key=lambda x: x[0])
#             best = (1.0, matches[-1][1], "last_occurrence_in_tail")

#     if not best:
#         return EvalResult(None, 0.0, True, "no_label_found", conflict=False)

#     score, label, reason = best

#     # Conflict detection (both labels appear somewhere)
#     has_true = _has_any(TRUE_TOKENS, text_noquotes)
#     has_false = _has_any(FALSE_TOKENS, text_noquotes)
#     conflict = has_true and has_false

#     # Confidence mapping
#     confidence = {3.0: 0.95, 2.0: 0.80, 1.0: 0.60}.get(score, 0.50)
#     needs_review = False

#     # If conflict but we only used weak evidence, flag review
#     if conflict and score <= 1.0:
#         needs_review = True
#         confidence = min(confidence, 0.55)

#     # If model explicitly says it’s re-evaluating, be cautious unless we used a strong cue
#     if re.search(r"wait[, ]+let'?s\s+re-?evaluate", text_noquotes, flags=re.IGNORECASE):
#         if score < 3.0:
#             needs_review = True

#     return EvalResult(label, confidence, needs_review, reason, conflict)


def evaluate_tf(response: str) -> EvalResult:
    raw = response or ""
    if not raw.strip():
        return EvalResult(None, 0.0, True, "empty_response", conflict=False)

    text_noquotes = _normalize(_strip_code_and_quotes(raw))

    # -------------------------------------------------
    # 0) First-occurrence decision (label vs abstain)
    # -------------------------------------------------
    first_label_pos = None
    first_label_value = None

    # find first true/false token
    for pat, lab in [(p, "true") for p in TRUE_TOKENS] + [(p, "false") for p in FALSE_TOKENS]:
        m = re.search(pat, text_noquotes, flags=re.IGNORECASE)
        if m:
            if first_label_pos is None or m.start() < first_label_pos:
                first_label_pos = m.start()
                first_label_value = lab

    # find first abstain
    first_abstain_pos = None
    for pat in ABSTAIN_PATTERNS:
        m = re.search(pat, text_noquotes, flags=re.IGNORECASE)
        if m:
            if first_abstain_pos is None or m.start() < first_abstain_pos:
                first_abstain_pos = m.start()

    # If both exist, take whichever appears first
    if first_label_pos is not None or first_abstain_pos is not None:
        if first_abstain_pos is not None and (
            first_label_pos is None or first_abstain_pos < first_label_pos
        ):
            return EvalResult(None, 0.0, True, "abstain_before_label", conflict=False)

        if first_label_pos is not None and (
            first_abstain_pos is None or first_label_pos < first_abstain_pos
        ):
            return EvalResult(first_label_value, 0.95, False, "first_explicit_label", conflict=False)

    # -------------------------------------------------
    # 1) Continue with your original structured logic
    # -------------------------------------------------
    best = None  # (score, label, reason)

    # Leading label
    m0 = re.match(
        r"^\s*[\*\s_`]*((?:true|false)|(?:صح|صحيح|صحيحة)|(?:خطأ|غلط|خاطئ|خاطئة))\b",
        text_noquotes,
        flags=re.IGNORECASE,
    )
    if m0:
        label = _match_label(m0.group(1))
        if label:
            best = (3.0, label, "leading_label")

    # Strong cues
    for cue in STRONG_CUES:
        for m in re.finditer(cue, text_noquotes, flags=re.IGNORECASE):
            window = text_noquotes[m.end():m.end() + 140]
            label = _match_label(window)
            if label:
                cand = (3.0, label, f"strong_cue:{cue}")
                best = max(best, cand, key=lambda x: x[0]) if best else cand

    # Standalone tail lines
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    tail_lines = lines[-25:]

    standalone_patterns = [
        (re.compile(r"^[\*\s_`]*true\s*[\.!\?]*[\*\s_`]*$", re.IGNORECASE), "true"),
        (re.compile(r"^[\*\s_`]*false\s*[\.!\?]*[\*\s_`]*$", re.IGNORECASE), "false"),
        (re.compile(r"^[\*\s_`]*صح\s*[\.!\?]*[\*\s_`]*$"), "true"),
        (re.compile(r"^[\*\s_`]*(خطأ|غلط|خاطئ|خاطئة)\s*[\.!\?]*[\*\s_`]*$"), "false"),
    ]

    if not best or best[0] < 3.0:
        for ln in reversed(tail_lines):
            ln_norm = _normalize(ln)
            for rgx, lab in standalone_patterns:
                if rgx.match(ln_norm):
                    best = (2.0, lab, "standalone_label_line")
                    break
            if best:
                break

    # Fallback tail search
    if not best:
        tail = text_noquotes[-450:]
        matches = []
        for pat in TRUE_TOKENS:
            for mm in re.finditer(pat, tail, flags=re.IGNORECASE):
                matches.append((mm.start(), "true"))
        for pat in FALSE_TOKENS:
            for mm in re.finditer(pat, tail, flags=re.IGNORECASE):
                matches.append((mm.start(), "false"))

        if matches:
            matches.sort(key=lambda x: x[0])
            best = (1.0, matches[-1][1], "last_occurrence_in_tail")

    if not best:
        return EvalResult(None, 0.0, True, "no_label_found", conflict=False)

    score, label, reason = best

    has_true = _has_any(TRUE_TOKENS, text_noquotes)
    has_false = _has_any(FALSE_TOKENS, text_noquotes)
    conflict = has_true and has_false

    confidence = {3.0: 0.95, 2.0: 0.80, 1.0: 0.60}.get(score, 0.50)
    needs_review = conflict and score <= 1.0

    return EvalResult(label, confidence, needs_review, reason, conflict)









def get_tf_flags(response: str):
    """
    Returns:
      has_true: bool
      has_false: bool
      is_unclear: bool
      is_conflict: bool   (both true+false appear somewhere in the text)
    """
    res = evaluate_tf(response)  # uses the full evaluator we wrote
    pred = res.pred  # "true" / "false" / None

    has_true = (pred == "true")
    has_false = (pred == "false")
    is_unclear = (pred is None)
    is_conflict = res.conflict


    

    return has_true, has_false, is_unclear, is_conflict






def _debug_one_response(resp: str):
    has_true, has_false, is_unclear, is_conflict = get_tf_flags(resp)
    if is_unclear: 
        final = "UNCLEAR"
    elif has_true:
        final = "TRUE"
    elif has_false:
        final = "FALSE"
    else:
        final = "UNKNOWN STATE" 
    print("DEBUG")
    print(f"Final", final)
    print(f"is_conflict_but_was_resolved", is_conflict and not is_unclear)

    
    
if __name__ == "__main__":
    
    #print(evaluate_tf("False. The image shows a man and a woman in formal attire with crowns on their heads, which suggests a more formal or ceremonial event rather than a casual family gathering. There are also other people in the background dressed in what looks like religious or official attire, further supporting the idea of a formal occasion. So, it's not likely to be a casual family gathering. What do you think about this? Do you have any other statements you want me to evaluate?"))
    print(evaluate_tf("صحيح."))
    #print(evaluate_tf('الإجابة النهائية: خطأ. شرح موجز لقرارك: العبارة تتحدث عن "الرجل" وهو مصطلح غير محدد في السياق. في سياق العباقرة، قد يشير إلى شخص ماهر أو متمكن، لكن بدون المزيد من المعلومات، لا يمكن تحديد ما إذا كان صحيحاً أو خاطئاً.'))
#     print(evaluate_tf("**خطأ.**").pred)   # should be "false"
#     print(evaluate_tf("**False.**").pred) # should be "false"
#     print(evaluate_tf("**صح.**").pred)    # should be "true"
#     print(evaluate_tf( "**خطأ.**\n\nالصورة تُظهر شخصين يرتديان زيًا عسكريًا (بما في ذلك الزي الكامو)، ويعملان على أجهزة إلكترونية (مثل اللوحيات أو الأجهزة المحمولة)، لكن **لا توجد دلائل واضحة على أن المكان هو مركز قيادة عسكري**.\n\nالإطار العام للصورة يبدو **مُستَلمًا أو مكتبًا مدنيًا أو بيئة تدريبية داخلية**، وليس مركز قيادة عسكريًا حقيقيًا، حيث:\n\n- الخلفية غير واضحة، لكنها تبدو كنوافذ أو جدران داخلية، وليس شاشات مراقبة أو خرائط أو معدات قيادة عسكرية.\n- لا توجد أي علامات على وجود نظام تحكم أو مراقبة عسكرية متقدمة.\n- التكوين البصري يشبه صورة ترويجية أو تدريبية، وليس صورة واقعية لمركز قيادة.\n\nإذًا، **الصورة لا تُظهر مركز قيادة عسكري بوضوح**، وبالتالي الجواب هو: **خطأ**."
#     ).pred)  # should be "false"
#     print(evaluate_tf("خطوات التفكير:  \n1 تُظهر الصورة ثرياً ضخماً معلقاً في مسجد، يتميز بتصميمه المعماري الإسلامي الدقيق، مع تفاصيل مزخرفة وعناصر معدنية لامعة  \n2 اللون السائد للثريا هو اللون الذهبي، مع وجود عناصر بيضاء أو شفافة (مثل الزجاج أو الألواح المضيئة) تُستخدم في أجزاء من الثريا  \n3 من المعلوم أن الثريات في المساجد الكبرى مثل المسجد النبوي أو الحرم المكي تُصنع من مواد فاخرة، وغالباً ما تُستخدم المعادن المطلية بالذهب أو الذهب الخالص في التفاصيل، بينما تُستخدم المعادن البيضاء (مثل الفضة أو الألمنيوم المطلي) في بعض الأجزاء لتعزيز التباين البصري والتألق  \n4 في الصورة، لا يمكن تحديد التركيب الكيميائي الدقيق للمواد من خلال الرؤية البصرية فقط، لكن من السياق العام (المسجد، وزارة الحج والعمرة، التصميم الفاخر) يُستنتج أن الثريا مصنوعة من مواد فاخرة تشمل الذهب والمعادن البيضاء (مثل الفضة أو المعادن المطلية)  \n5 العبارة تقول \"مصنوعة من الذهب والمعادن البيضاء\"، وهي عبارة عامة تُستخدم في وصف مثل هذه الثريات، ولا تُنفي وجود مواد أخرى (مثل الزجاج أو البلاستيك المضيء)، لكنها تركز على المواد المعدنية البارزة  \n6 بناءً على السياق والتصميم، فإن وصف الثريا بأنها مصنوعة من الذهب والمعادن البيضاء هو وصف معقول ودقيق من حيث التصميم والقيمة الفنية\n\nالإجابة النهائية هي <صحيح>"
# ).pred)  # should be "true"
#     print(evaluate_tf("خطوات التفكير:  \n1 تُظهر الصورة رجلاً يعزف على آلة موسيقية ذات شكل مميز، تشبه العود أو آلة مماثلة من الموسيقى الشرقية أو التقليدية، مع وجود تفاصيل زخرفية على جسم الآلة  \n2 يرتدي العازف نظارات وملابس رسمية نسبيًا، ويجلس أمام ميكروفون، مما يشير إلى أن العرض يتم في مكان مخصص للعروض الموسيقية  \n3 في الزاوية العلوية اليسرى، يوجد شعار يحمل عبارة \"summer sessions live from grand junction\"، مما يشير إلى أن الحدث هو عرض مباشر ضمن سلسلة عروض موسيقية، وليس بالضرورة عرضًا تقليديًا بمعنى التراث الشعبي فقط  \n4 رغم أن الآلة الموسيقية تُعتبر تقليدية، إلا أن السياق العام (ميكروفون، إضاءة مسرحية، شعار حدث حديث) يشير إلى أن العرض قد يكون جزءًا من مهرجان أو سلسلة عروض حديثة تضم موسيقى تقليدية أو معاصرة  \n5 لا يمكن تحديد طبيعة الموسيقى (هل هي تقليدية بالكامل أم معاصرة أو مدمجة) من الصورة وحدها، لكن السياق يشير إلى أن الحدث هو عرض موسيقي حديث يضم عناصر تقليدية  \n6 العبارة تقول \"عرض موسيقي تقليدي\"، وهذا قد يكون مبالغة أو غير دقيق، لأن السياق يوحي بحدث حديث يضم موسيقى تقليدية، وليس بالضرورة عرضًا \"تقليديًا\" بالمعنى الكامل (مثل عرض شعبي أو تراثي في مكان تقليدي)  \n\nالإجابة النهائية هي خطأ").pred)
#     print(evaluate_tf("خطوات التفكير:  \n1 تُظهر الصورة أربعة لاعبين يرتدون زيًا رياضيًا موحدًا باللون الأخضر والأسود، ويحمل كل لاعب رقمًا على قميصه (10، 25، 5، ورقم غير واضح للرابع)، وهو ما يُعد مميزًا لزي فرق كرة القدم  \n2 يظهر على القمصان شعار يشبه شعار الاتحاد السعودي لكرة القدم، بالإضافة إلى علامة \"adidas\"، وهي شركة رياضية تُنتج معدات كرة القدم بشكل واسع  \n3 يُلاحظ أن اللاعبين يرتدون شراشف رياضية سوداء، وقبعات رياضية بيضاء، وحذاء رياضي مناسب للعب على العشب (أحذية كرة قدم)  \n4 في المقدمة، توجد عوائق صفراء (hurdles) ومخروط أخضر، وهي أدوات تُستخدم في تدريبات كرة القدم لتحسين التوازن، السرعة، والمناورة  \n5 المكان هو ملعب عشب اصطناعي، وهو بيئة شائعة لتدريبات كرة القدم  \n6 وضعية اللاعبين (مثلاً اللاعب رقم 5 في وضعية تمرين توازن أو تدريب على التحرك الجانبي) تتوافق مع تدريبات كرة القدم  \n\nبناءً على هذه العناصر، فإن كل ما يظهر في الصورة يتوافق مع تدريبات كرة القدم، وليس رياضة أخرى\n\nالإجابة النهائية هي <صحيح>").pred)
#     print(evaluate_tf("خطوات التفكير:  \n1 تُظهر الصورة جزءًا من طقم طبول (drum kit) من زاوية قريبة، مع تركيز على طبول صغيرة مثل السناير (snare drum) والطبلة العلوية (tom-tom)، بالإضافة إلى يد العازف وعصي الطبول  \n2 السناير (snare drum) هو الطبل الصغير المركزي في الصورة، ويظهر بوضوح في الجزء السفلي من الصورة، مع عصي الطبول التي تلامسه  \n3 من حيث الوضوح والتركيز البصري، فإن السناير هو الجزء الأكثر وضوحًا وتفاصيلًا في الصورة، مقارنة بالطبلة العلوية التي تظهر في الخلفية بشكل أقل وضوحًا، أو بقية الطقم  \n4 كما أن السناير يشغل مساحة كبيرة في المقدمة، ويُعتبر محور التركيز البصري للصورة، بينما تظهر أجزاء أخرى مثل الكونترباس أو الطبلة الكبيرة بشكل غير واضح أو مُحجب  \n5 العبارة تقول \"أكتر جزء باين\" – أي \"أكتر جزء بارز أو واضح\" – وهذا يتوافق مع السناير، لأنه يظهر بوضوح وتفاصيل، ويُعتبر الجزء المركزي في الصورة  \n\nالإجابة النهائية هي صحيح").pred)
#     # examples = [
#     #     "**False.**\n\n... Therefore, the statement is **True**.",
    #     "Conclusion:\n\nThe statement is **False**.",
    #     "لذلك، لا يمكننا الجزم بصحة أو خطأ العبارة بناءً على الصورة فقط.",
    #     "الإجابة الصحيحة هي: **خطأ**.",
    #     "خطأ\n\nالعبارة غير صحيحة في الصورة، يبدو أن هناك مادة بيضاء حول قرص الفحم، والتي تشبه الملح، ولكنها ليست بالضرورة الملح في الاحتفاظ بقرص الفحم، قد يتم استخدام مادة أخرى مثل الرمال أو الرماد أو حتى الطين، ولكن الملح ليس هو الخيار الأكثر شيوعًا", 
#     "**خطأ.**\n\nالصورة تُظهر شخصين يرتديان زيًا عسكريًا (بما في ذلك الزي الكامو)، ويعملان على أجهزة إلكترونية (مثل اللوحيات أو الأجهزة المحمولة)، لكن **لا توجد دلائل واضحة على أن المكان هو مركز قيادة عسكري**.\n\nالإطار العام للصورة يبدو **مُستَلمًا أو مكتبًا مدنيًا أو بيئة تدريبية داخلية**، وليس مركز قيادة عسكريًا حقيقيًا، حيث:\n\n- الخلفية غير واضحة، لكنها تبدو كنوافذ أو جدران داخلية، وليس شاشات مراقبة أو خرائط أو معدات قيادة عسكرية.\n- لا توجد أي علامات على وجود نظام تحكم أو مراقبة عسكرية متقدمة.\n- التكوين البصري يشبه صورة ترويجية أو تدريبية، وليس صورة واقعية لمركز قيادة.\n\nإذًا، **الصورة لا تُظهر مركز قيادة عسكري بوضوح**، وبالتالي الجواب هو: **خطأ**."

    # ]
    # for resp in examples:
    #     print("===================================")
    #     print("Response:")
    #     print(resp)
    #     _debug_one_response(resp)
    #     print("===================================")