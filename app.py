
import streamlit as st
import requests
import json
import re
import pandas as pd
import zipfile
import io
import time

st.set_page_config(
    page_title="AI Promo Finder",
    page_icon="🎬",
    layout="wide"
)

MODEL = "gemini-3.1-flash-lite"

st.title("🎬 AI Promo Finder")
st.write(
    "Upload screenplay files to identify, validate and rank "
    "short-form promotional clips."
)

# --------------------------------------------------
# API KEY
# --------------------------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""

if not api_key:
    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def call_gemini(prompt, api_key, temperature=0.2):

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{MODEL}:generateContent?key={api_key}"
    )

    response = requests.post(
        url,
        json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


def parse_json(raw):

    clean = raw.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    first = clean.find("{")
    last = clean.rfind("}")

    if first == -1 or last == -1:
        raise ValueError("No valid JSON found")

    return json.loads(clean[first:last + 1])


def to_seconds(timestamp):

    minutes, seconds = timestamp.split(":")
    return int(minutes) * 60 + float(seconds)


def get_all_timestamps(text):

    pattern = r"(?<!\d)(\d{2}:\d{2}(?:\.\d{1,2})?)(?!\d)"

    return set(
        round(to_seconds(t), 2)
        for t in re.findall(pattern, text)
    )


def get_episode_number(filename, text, fallback):

    match = re.search(
        r"(?:episode|ep)[ _-]*0*(\d+)",
        filename,
        re.IGNORECASE
    )

    if not match:
        match = re.search(
            r"EPISODE NUMBER\s*[:\-]?\s*(\d+)",
            text,
            re.IGNORECASE
        )

    return int(match.group(1)) if match else fallback


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload screenplay TXT files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} screenplay files uploaded.")

# --------------------------------------------------
# PROCESS SCREENPLAYS
# --------------------------------------------------

if st.button("Analyze Screenplays", type="primary"):

    if not api_key:
        st.error("Please enter your Gemini API key.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one screenplay.")
        st.stop()

    all_candidates = []
    raw_outputs = {}
    errors = []

    progress = st.progress(0)
    status = st.empty()

    # Generate promo candidates episode by episode

    for file_index, uploaded_file in enumerate(uploaded_files):

        try:
            screenplay = uploaded_file.getvalue().decode(
                "utf-8", errors="replace"
            )

            episode_no = get_episode_number(
                uploaded_file.name,
                screenplay,
                file_index + 1
            )

            status.write(
                f"Analyzing Episode {episode_no}..."
            )

            prompt = f"""
You are an expert short-form promotional video editor.

Analyze the following screenplay and identify up to 3
compelling promotional clips.

Each clip MUST be between 20 and 60 seconds.

Prioritize:
- Strong opening hooks
- Betrayal, conflict, mystery or emotional tension
- A compelling cliffhanger
- Clear storytelling for a new viewer

IMPORTANT:
1. Use only timestamps present in the screenplay.
2. Start and end must match actual event boundaries.
3. Do not invent dialogue, events or timestamps.
4. Return up to 3 distinct candidates.
5. Return ONLY valid JSON.

Required format:
{{
  "candidates": [
    {{
      "episode": {episode_no},
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "hook": "Opening hook",
      "conflict": "Main conflict",
      "cliffhanger": "Ending moment",
      "why_it_works": "Why the clip could engage viewers"
    }}
  ]
}}

SCREENPLAY:
{screenplay}
"""

            raw = call_gemini(prompt, api_key)
            raw_outputs[uploaded_file.name] = raw

            data = parse_json(raw)

            valid_timestamps = get_all_timestamps(screenplay)

            for clip in data.get("candidates", []):

                start = clip["start"]
                end = clip["end"]

                duration = (
                    to_seconds(end) - to_seconds(start)
                )

                start_match = round(to_seconds(start), 2) in valid_timestamps

                end_match = round(to_seconds(end), 2) in valid_timestamps

                all_candidates.append({
                    "episode": episode_no,
                    "start": start,
                    "end": end,
                    "duration": round(duration, 2),
                    "valid_duration": 20 <= duration <= 60,
                    "timestamp_verified": (
                        start_match and end_match
                    ),
                    "hook": clip.get("hook", ""),
                    "conflict": clip.get("conflict", ""),
                    "cliffhanger": clip.get("cliffhanger", ""),
                    "why_it_works": clip.get("why_it_works", "")
                })

        except Exception as e:
            errors.append(
                f"{uploaded_file.name}: {str(e)}"
            )

        progress.progress(
            (file_index + 1) / len(uploaded_files)
        )

        time.sleep(1)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    candidates_df = pd.DataFrame(all_candidates)

    if candidates_df.empty:
        st.error("No promo candidates were generated.")
        st.write(errors)
        st.stop()

    candidates_df["candidate_id"] = range(
        1, len(candidates_df) + 1
    )

    valid_df = candidates_df[
        candidates_df["valid_duration"]
        & candidates_df["timestamp_verified"]
    ].copy()

    st.subheader("Analysis Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Episodes uploaded", len(uploaded_files))
    c2.metric("Candidates generated", len(candidates_df))
    c3.metric("Valid duration", len(
        candidates_df[candidates_df["valid_duration"]]
    ))
    c4.metric("Verified candidates", len(valid_df))

    # --------------------------------------------------
    # EDITORIAL EVALUATION
    # --------------------------------------------------

    if valid_df.empty:
        st.warning("No candidates passed validation.")
        st.stop()

    status.write("Evaluating valid candidates...")

    judge_results = []

    for batch_start in range(0, len(valid_df), 20):

        batch = valid_df.iloc[
            batch_start:batch_start + 20
        ]

        candidate_data = batch[[
            "candidate_id",
            "episode",
            "duration",
            "hook",
            "conflict",
            "cliffhanger",
            "why_it_works"
        ]].to_dict(orient="records")

        judge_prompt = f"""
You are a professional short-form audio drama promo editor.

Evaluate each candidate from 1 to 5 on:

- hook_score: Strength of opening hook
- conflict_score: Strength of central conflict
- cliffhanger_score: Curiosity created by the ending
- clarity_score: How easily a new viewer understands it
- originality_score: How distinctive the story moment is

Do not reward dramatic keywords alone.
Do not invent events or assume visual information.

Return ONLY valid JSON:
{{
  "evaluations": [
    {{
      "candidate_id": 1,
      "hook_score": 4,
      "conflict_score": 5,
      "cliffhanger_score": 4,
      "clarity_score": 4,
      "originality_score": 5,
      "reason": "Short explanation"
    }}
  ]
}}

Evaluate every candidate exactly once.

CANDIDATES:
{json.dumps(candidate_data, ensure_ascii=False)}
"""

        raw_judgement = call_gemini(
            judge_prompt,
            api_key,
            temperature=0.1
        )

        evaluations = parse_json(
            raw_judgement
        )["evaluations"]

        judge_results.extend(evaluations)

    judge_df = pd.DataFrame(judge_results)

    # --------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------

    final_df = valid_df.merge(
        judge_df,
        on="candidate_id",
        how="inner"
    )

    final_df["editorial_score"] = (
        final_df["hook_score"] * 0.25
        + final_df["conflict_score"] * 0.25
        + final_df["cliffhanger_score"] * 0.25
        + final_df["clarity_score"] * 0.15
        + final_df["originality_score"] * 0.10
    ) * 20

    final_df = final_df.sort_values(
        "editorial_score",
        ascending=False
    ).reset_index(drop=True)

    final_df.insert(
        0,
        "rank",
        range(1, len(final_df) + 1)
    )

    status.success("Analysis completed!")

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    st.subheader("🏆 Ranked Promo Candidates")

    st.dataframe(
        final_df[[
            "rank",
            "episode",
            "start",
            "end",
            "duration",
            "editorial_score",
            "hook",
            "conflict",
            "cliffhanger",
            "reason"
        ]],
        use_container_width=True
    )

    st.subheader("Top 5 Recommendations")

    for _, row in final_df.head(5).iterrows():

        with st.expander(
            f"Rank {row['rank']} | Episode {row['episode']} "
            f"| Score: {row['editorial_score']:.0f}/100"
        ):

            st.write("**Timing:**", row["start"], "-", row["end"])
            st.write("**Hook:**", row["hook"])
            st.write("**Conflict:**", row["conflict"])
            st.write("**Cliffhanger:**", row["cliffhanger"])
            st.write("**Why it works:**", row["reason"])

    # --------------------------------------------------
    # DOWNLOAD OUTPUTS
    # --------------------------------------------------

    st.subheader("Download Results")

    st.download_button(
        "Download ranked promos CSV",
        final_df.to_csv(index=False),
        "ranked_promos.csv",
        "text/csv"
    )

    st.download_button(
        "Download all candidates CSV",
        candidates_df.to_csv(index=False),
        "all_candidates.csv",
        "text/csv"
    )

    st.download_button(
        "Download raw AI outputs",
        json.dumps(raw_outputs, indent=2),
        "raw_ai_outputs.json",
        "application/json"
    )

    if errors:
        st.warning("Some files could not be processed:")
        for error in errors:
            st.write(error)
