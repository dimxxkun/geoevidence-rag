"""Local analyst workbench: run with `uv run streamlit run app.py`."""

import os
from datetime import date
from pathlib import Path

import pydeck as pdk
import streamlit as st

from geoevidence.engine import Engine
from geoevidence.models import Query
from geoevidence.store import load_index

st.set_page_config(page_title="GeoEvidence | Spatial evidence desk", page_icon="🌐", layout="wide")


@st.cache_resource
def open_engine(path: str, modified: int, generator: str):
    return Engine.open(Path(path), generator)


@st.cache_data
def document_rows(path: str, modified: int):
    index = load_index(Path(path))
    documents = {chunk.document.id: chunk.document for chunk in index.chunks}
    return index.metadata, [
        {
            "id": doc.id,
            "title": doc.title,
            "lon": doc.location.lon,
            "lat": doc.location.lat,
            "topic": doc.topic,
        }
        for doc in documents.values()
    ]


st.caption("GEOEVIDENCE / SPATIAL EVIDENCE DESK")
st.title("Ask a place. Inspect the evidence.")
st.write(
    "Search geotagged briefs with explicit geographic boundaries and traceable source passages."
)
st.info(
    "Demo records are fictional. Coordinates mark approximate exercise locations, "
    "not real assets or verified conditions."
)
path = Path(os.environ.get("GEOEVIDENCE_INDEX", "artifacts/demo.sqlite"))
if not path.is_file():
    st.warning("Build the demo index first: geoevidence index data/demo.geojson")
    st.stop()
modified = path.stat().st_mtime_ns
metadata, rows = document_rows(str(path), modified)
neural = metadata["encoder"] != "lexical"

with st.sidebar:
    st.header("Define your search area")
    spatial = st.toggle("Limit to a radius", value=True)
    lon = st.number_input("Center longitude", -180.0, 180.0, 3.3792, format="%.4f")
    lat = st.number_input("Center latitude", -90.0, 90.0, 6.5244, format="%.4f")
    radius = st.slider("Radius (km)", 1, 100, 8)
    topics = ["All topics", *sorted({row["topic"] for row in rows})]
    topic = st.selectbox("Topic", topics, index=topics.index("flood") if "flood" in topics else 0)
    use_date = st.toggle("Exclude older briefs", value=True)
    start = st.date_input("Published from", date(2026, 1, 1))
    mode = st.selectbox("Search method", ["hybrid", "dense", "lexical"] if neural else ["lexical"])
    generator = st.selectbox("Answer style", ["extractive", "flan"] if neural else ["extractive"])
    st.caption(
        "Extractive quotes passages verbatim. FLAN produces an unverified model summary "
        "from the displayed context."
    )

question = st.text_input("Your question", "What drainage maintenance is recommended?")
if st.button("Find evidence", type="primary"):
    filters = {}
    if spatial:
        filters["radius"] = {"center": {"lon": lon, "lat": lat}, "km": radius}
    if topic != "All topics":
        filters["topic"] = topic
    if use_date:
        filters["published_from"] = start
    try:
        with st.spinner("Retrieving evidence..."):
            engine = open_engine(str(path), modified, generator)
            st.session_state["result"] = engine.ask(
                Query(question=question, filters=filters, mode=mode)
            )
            st.session_state["submitted_filters"] = filters
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))

result = st.session_state.get("result")
selected = {item.document_id for item in result.evidence} if result else set()
for row in rows:
    row["color"] = [20, 184, 166, 230] if row["id"] in selected else [100, 116, 139, 150]
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Document anchors")
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=lat, longitude=lon, zoom=10 if spatial else 4
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    rows,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=350,
                    radius_min_pixels=6,
                    pickable=True,
                )
            ],
            tooltip={"text": "{title}\n{topic}"},
        )
    )
    st.caption(
        "Teal = retrieved evidence from the last submitted question. Grey = other document "
        "anchors. Basemap uses online tiles; CLI/API work without tiles."
    )
with right:
    st.subheader("Evidence-linked answer")
    if result:
        st.caption(f"Last submitted question: {result.question}")
        if result.grounding == "unverified_generation":
            st.warning(
                "Model-generated summary. Evidence labels identify supplied context; "
                "they do not verify every claim."
            )
        st.write(result.answer)
        st.caption(
            f"{result.eligible_documents} eligible documents · {len(result.evidence)} passages · "
            f"{result.elapsed_ms:.1f} ms including generation"
        )
        for item in result.evidence:
            with st.expander(f"[{item.citation}] {item.title}"):
                st.write(item.quote)
                st.caption(f"{item.source} · {item.published} · characters {item.start}:{item.end}")
                st.json(
                    {
                        "location": item.location.model_dump(),
                        "distance_km": item.distance_km,
                        "rank_score": item.score,
                        "cosine_similarity": item.cosine_similarity,
                        "lexical_score": item.lexical_score,
                    }
                )
        with st.expander("Submitted filters"):
            st.json(st.session_state["submitted_filters"])
        st.download_button(
            "Download answer + evidence",
            result.model_dump_json(indent=2),
            file_name="geoevidence-answer.json",
            mime="application/json",
        )
    else:
        st.write(
            "Choose an area and submit a question. The exact supporting text will appear here."
        )
st.caption(
    f"Snapshot {metadata['fingerprint'][:12]} · {metadata['document_count']} documents · "
    f"{metadata['chunk_count']} chunks"
)
