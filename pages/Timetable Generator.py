# timetable_app.py  –  complete & tested 2025-04-30
import os, json, copy, datetime as dt
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────────────────────────
# 🔑  Load Groq API Key
# ─────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found – add it to your .env or env vars")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────
# 🌐  Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Timetable Generator", layout="wide")
st.title("📘 AI Timetable Generator (Groq + LLaMA-4)")

DAY_CHOICES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIOD_FMT  = "Period {idx}"

def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Pull the outermost JSON object from `text`, even if there's extra junk before
    or after it. Raises ValueError if no well-formed JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    brace_count = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1

        # once we've closed all opened braces, that's the end of the JSON
        if brace_count == 0:
            end = i
            break
    else:
        raise ValueError("JSON object not closed properly in LLM response")

    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # re-raise with some context
        raise ValueError(f"Failed to parse JSON block: {e}\n---\n{json_str}") from e

# ─────────────────────────────────────────────────────────────
# 1️⃣  SCHOOL TAB
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["School", "Classes", "Rooms", "Teachers", "Generate"])
with tabs[0]:
    st.header("School")
    school_days = st.multiselect("Working days", DAY_CHOICES, default=DAY_CHOICES)

    c1, c2, c3 = st.columns(3)
    with c1:
        num_periods = st.slider("Periods / day (excluding lunch)", 4, 10, 6)
    with c2:
        period_len  = st.number_input("Minutes per period", 30, 90, 40)
    with c3:
        gap_len     = st.number_input("Gap between periods (min)", 0, 20, 5)

    base_time = dt.datetime.combine(
        dt.date.today(),
        st.time_input("🕗 School starts", dt.time(8, 0))
    )

    periods: List[Dict[str,str]] = []
    st.subheader("Period times")
    for idx in range(num_periods):
        def_start = base_time + dt.timedelta(minutes=idx*(period_len+gap_len))
        def_end   = def_start + dt.timedelta(minutes=period_len)
        s_col, e_col = st.columns(2)
        with s_col:
            s_time = st.time_input(f"Start {PERIOD_FMT.format(idx=idx+1)}",
                                   def_start.time(), key=f"s_{idx}")
        with e_col:
            e_time = st.time_input(f"End   {PERIOD_FMT.format(idx=idx+1)}",
                                   def_end.time(),   key=f"e_{idx}")
        periods.append({
            "label": PERIOD_FMT.format(idx=idx+1),
            "time":  f"{s_time.strftime('%H:%M')}-{e_time.strftime('%H:%M')}"
        })

    add_lunch  = st.checkbox("🍴 Add lunch break", True)
    lunch_after = st.slider("Insert lunch after period #", 1, num_periods-1, 3) if add_lunch else None

# ─────────────────────────────────────────────────────────────
# 2️⃣  CLASSES TAB
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Classes")
    default_cls = [f"Class {i}" for i in range(6, 13)]
    class_names = st.multiselect("Select classes", default_cls, default=["Class 9"])

    extra_cls = st.text_input("Add classes (comma-sep)")
    if extra_cls:
        class_names += [c.strip() for c in extra_cls.split(",") if c.strip()]
        class_names  = sorted(set(class_names))

    default_subs = ["English","Math","Science","Urdu","Islamiat",
                    "Computer","Physics","Chemistry","Biology"]
    subjects_pool = default_subs.copy()
    extra_subs = st.text_input("Add subjects (comma-sep)")
    if extra_subs:
        subjects_pool += [s.strip() for s in extra_subs.split(",") if s.strip()]
        subjects_pool  = sorted(set(subjects_pool))

    lesson_plan: Dict[str,Any] = {}
    st.markdown("---")
    for cname in class_names:
        with st.expander(f"📚 {cname}"):
            strength = st.number_input("👥 Size", 10, 100, 30, key=f"{cname}_size")
            sel_subs = st.multiselect("Subjects", subjects_pool,
                                      default=subjects_pool[:6], key=f"{cname}_subs")
            freq = {s: st.slider(f"{s} periods/week", 1, 10, 4,
                                 key=f"{cname}_{s}_freq") for s in sel_subs}
            lesson_plan[cname] = {"strength": strength, "subjects": freq}

# ─────────────────────────────────────────────────────────────
# 3️⃣  ROOMS TAB
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Rooms")
    num_rooms = st.slider("Room count", 1, 30, 8)
    rooms: List[Dict[str,Any]] = []
    for i in range(num_rooms):
        c1, c2 = st.columns(2)
        with c1:
            rname = st.text_input(f"Room {i+1}", value=f"Room-{i+1}", key=f"room_{i}")
        with c2:
            cap   = st.number_input("Capacity", 10, 160, 40, key=f"cap_{i}")
        rooms.append({"name": rname, "capacity": cap})

# ─────────────────────────────────────────────────────────────
# 4️⃣  TEACHERS TAB
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Teachers")
    num_teachers = st.slider("Teacher count", 1, 50, 10)
    teachers: Dict[str,Any] = {}

    all_slots = [f"{d[:3]}{p+1}" for d in school_days for p in range(num_periods)]
    for i in range(num_teachers):
        with st.expander(f"👩‍🏫 Teacher {i+1}"):
            tname = st.text_input("Name", value=f"Teacher-{i+1}", key=f"tname_{i}")
            teaches = st.multiselect("Subjects", subjects_pool,
                                     default=[subjects_pool[i % len(subjects_pool)]],
                                     key=f"teach_{i}")
            unavail = st.multiselect("Unavailable (Mon1…)", all_slots, key=f"unavail_{i}")
            teachers[tname] = {"subjects": teaches, "unavailable": unavail}

# ─────────────────────────────────────────────────────────────
# 5️⃣  GENERATE TAB  (validate + repair)
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Generate Timetable")
    if st.button("🧠 Generate"):
        if not school_days or not class_names:
            st.error("Need at least one working day and one class.")
            st.stop()

        # Normalise class names
        class_names = [c.replace("\u00A0"," ").strip() for c in class_names]

        # Build teacher availability once
        for t, info in teachers.items():
            merged = set(info["unavailable"])
            info["availability"] = sorted([s for s in all_slots if s not in merged])

        # Conflict detector
        def clashes(tt: Dict[str,Any]) -> List[str]:
            seen_t, seen_r, out = {}, {}, []
            for cls, dmap in tt.items():
                for day, pmap in dmap.items():
                    for per, inf in pmap.items():
                        if inf.get("subject") == "🍴 Lunch": continue
                        slot = f"{day[:3]}{per.replace('Period','')}"
                        t, r = inf["teacher"], inf["room"]
                        if (slot,t) in seen_t:
                            out.append(f"{t} in {slot} (classes {seen_t[(slot,t)]}, {cls})")
                        seen_t[(slot,t)] = cls
                        if (slot,r) in seen_r:
                            out.append(f"{r} in {slot} (classes {seen_r[(slot,r)]}, {cls})")
                        seen_r[(slot,r)] = cls
            return out

        # System prompt
        system_prompt = """
You are an expert school timetable generator.
Obey *all* constraints:

• A teacher or room may appear in **only one class** at the same time in the same day/period.
• A teacher may only teach subjects they are assigned to.
• A room may only be used for one class at a time.
• Respect any teacher's unavailable slots.
• Create random timetable for each class.
• Each class has a unique set of subjects and periods/week.
• If I point out conflicts, regenerate only the specified class to fix them,
  changing as little as possible.

Return JSON exactly like:
{
  "timetable": {
    "Class X": {
      "Monday": {
        "Period1": {"subject": "...", "teacher": "...", "room": "..."},
        ...
      },
      ...
    }
  }
}
"""
        messages = [{"role":"system","content":system_prompt}]
        full_tt: Dict[str,Any] = {}

        # ----- main loop over classes -----
        for cname in class_names:
            for attempt in range(1,4):
                st.info(f"{cname} – attempt {attempt}")

                payload = {
                    "school_days": school_days,
                    "periods": periods,
                    "classes": {cname: lesson_plan[cname]},
                    "teachers": teachers,
                    "rooms": rooms,
                    "already_scheduled": full_tt
                }
                if add_lunch: payload["lunch_break_after"] = lunch_after

                messages.append({"role":"user",
                                 "content":json.dumps(payload,indent=2)})
                resp = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=messages,
                    temperature=0,
                )
                assistant_msg = resp.choices[0].message.content
                messages.append({"role":"assistant","content":assistant_msg})

                block = extract_json_block(assistant_msg).get("timetable", {})
                key   = next((k for k in block
                              if k.replace(" ","").lower()==cname.replace(" ","").lower()), None)
                if not key:
                    st.error(f"LLM did not return a timetable for {cname}")
                    break

                cand_tt = copy.deepcopy(full_tt)
                cand_tt[cname] = block[key]

                bad = clashes(cand_tt)
                if not bad:
                    full_tt = cand_tt   # accepted!
                    break

                if attempt < 3:
                    msg = ("Conflicts found for **{cls}**:\n"
                           + "\n".join(f"- {c}" for c in bad[:10])
                           + "\nRegenerate only that class with no conflicts.")
                    messages.append({"role":"user","content":msg.format(cls=cname)})
                else:
                    st.warning(f"Keeping best-effort timetable for {cname} after 3 failures.")
                    full_tt = cand_tt

        # ----- build display-period order -----
        display_periods = []
        for i,p in enumerate(periods,start=1):
            display_periods.append(p)
            if add_lunch and i == lunch_after:
                display_periods.append({"label":"🍴 Lunch","time":""})

        idx_labels = [f"{p['label']}{('\n'+p['time']) if p['time'] else ''}"
                      for p in display_periods]

        # ----- render each class -----
        for cname, day_map in full_tt.items():
            st.subheader(f"📅 {cname}")
            df = pd.DataFrame(index=idx_labels, columns=school_days)

            for day in school_days:
                for p in display_periods:
                    row = f"{p['label']}{('\n'+p['time']) if p['time'] else ''}"
                    if p["label"] == "🍴 Lunch":
                        df.at[row,day] = "🍴 Lunch"
                        continue
                    num = int(p["label"].split()[1])
                    ent = day_map.get(day, {}).get(f"Period{num}", {})
                    if ent:
                        df.at[row,day] = (f"{ent['subject']}\n"
                                          f"({ent['teacher']})\n"
                                          f"{ent['room']}")
                    else:
                        df.at[row,day] = "—"

            st.dataframe(df, use_container_width=True,
                         height=min(600, 60*len(idx_labels)))
            st.download_button(
                f"💾 Download {cname} CSV",
                df.to_csv().encode("utf-8"),
                f"{cname}_timetable.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────
# ℹ️  Sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.write(
    "This version validates each timetable against teacher/room clashes and "
    "asks LLaMA-4 to repair any conflicts (up to 3 retries per class)."
)
