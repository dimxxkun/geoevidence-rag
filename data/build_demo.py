"""Author the reproducible fictional corpus; no scraped or confidential records."""

import json
from pathlib import Path

# Coordinates anchor fictional briefs near real places. They are not surveyed assets.
RECORDS = [
    (
        "lag-drain",
        "Mainland drainage inspection",
        3.3792,
        6.5244,
        "flood",
        "2026-06-12",
        "The fictional mainland inspection found blocked culverts along the canal approach. "
        "Crews should remove plastic waste and accumulated sediment before the wet-season"
        " exercise. "
        "The maintenance plan prioritizes clearing the inlet grates, photographing each "
        "obstruction, "
        "and logging downstream water levels after cleaning. No pump purchase was approved. "
        "This brief recommends a follow-up inspection within seven days of the exercise. "
        "The record describes a training scenario and does not establish actual flood "
        "conditions in Lagos. "
        "Inspectors should attach before-and-after photographs to the same work order so "
        "the follow-up "
        "team can distinguish repeat blockage from incomplete clearance. The exercise "
        "supervisor signs "
        "off only after the inlet and outlet have both been checked.",
    ),
    (
        "lag-pump",
        "Island pump readiness",
        3.4300,
        6.4300,
        "flood",
        "2026-06-14",
        "The fictional island response depot has two portable pumps reserved for "
        "stormwater exercises. "
        "Operators must test hoses and fuel supply before deployment. The staging point "
        "is inland from "
        "the shoreline. The exercise plan assigns a safety observer to each pump team and"
        " prohibits "
        "pumping into an already obstructed drain. This is a preparedness inventory, not "
        "a live emergency notice.",
    ),
    (
        "lag-drain-old",
        "Archived mainland drainage inspection",
        3.3790,
        6.5240,
        "flood",
        "2025-04-10",
        "An earlier fictional drainage inspection recommended purchasing a fixed pump and widening "
        "the canal inlet. The proposal was exploratory and had no funding decision. This "
        "archived brief "
        "predates the current maintenance exercise and must not be substituted for the June 2026 "
        "clearance plan. Dates matter because historical recommendations can conflict "
        "with current instructions.",
    ),
    (
        "abu-drain",
        "Central district drainage inspection",
        7.4893,
        9.0579,
        "flood",
        "2026-06-12",
        "The fictional central district drainage team should replace a collapsed concrete culvert. "
        "Removing surface litter alone will not restore the crossing because a structural"
        " panel has failed. "
        "The exercise budget covers a temporary bypass and an engineering survey. The "
        "replacement is "
        "scheduled after survey approval. These actions belong to the Abuja exercise "
        "area, not the mainland Lagos plan.",
    ),
    (
        "ph-drain",
        "Southern drainage access",
        7.0134,
        4.8156,
        "flood",
        "2026-06-11",
        "The fictional southern drainage inspection concerns access to a tidal outfall. "
        "Crews should "
        "check the flap gate at low tide and clear vegetation around the maintenance "
        "platform. A mobile "
        "barrier marks the slippery approach. The exercise coordinator must confirm tide "
        "timing before "
        "dispatch. Replacing an inland culvert is outside this record's work order.",
    ),
    (
        "lag-feeder",
        "Mainland feeder reliability",
        3.3550,
        6.5500,
        "electricity",
        "2026-07-02",
        "The fictional mainland electricity feeder has three recurring voltage-drop reports during "
        "the evening exercise. The engineering recommendation is to rebalance phase loads before "
        "considering transformer replacement. Meter readings must be captured at both "
        "ends of the feeder. "
        "The planning worksheet assigns phase balancing to the first maintenance window and leaves "
        "capital upgrades pending measurement. No actual customer or utility network is "
        "represented.",
    ),
    (
        "abu-feeder",
        "Northern feeder reliability",
        7.4500,
        9.0900,
        "electricity",
        "2026-07-03",
        "The fictional northern feeder inspection found cracked insulators near the "
        "training substation. "
        "The recommended work is insulator replacement under an approved isolation "
        "procedure. Voltage "
        "balancing is not the primary fault in this brief. The maintenance team records "
        "component serial "
        "numbers and photographs after replacement. The location anchor is approximate "
        "and is not an actual asset.",
    ),
    (
        "lag-water",
        "Mainland drinking-water maintenance",
        3.3900,
        6.5350,
        "water",
        "2026-05-20",
        "The fictional water maintenance team plans to replace leaking service valves "
        "near the training "
        "junction. After repairs, operators flush the affected section and collect "
        "water-quality samples. "
        "The brief does not report contamination or certify water as safe. Any real "
        "drinking-water decision "
        "requires current accredited testing. The exercise inventory lists six "
        "replacement valves and two sampling kits.",
    ),
    (
        "abu-water",
        "Reservoir instrumentation exercise",
        7.4200,
        9.0400,
        "water",
        "2026-05-21",
        "The fictional reservoir exercise focuses on level-sensor calibration. "
        "Technicians compare the "
        "telemetry reading with a manual staff-gauge reading before changing alarm "
        "thresholds. The plan "
        "provides a reference checklist and requires a second operator to verify the "
        "offset. Valve replacement "
        "is not part of this work order. This record is a training brief, not a reservoir"
        " operating instruction.",
    ),
    (
        "lag-heat",
        "Mainland shade corridor",
        3.3700,
        6.5100,
        "heat",
        "2026-03-15",
        "The fictional heat adaptation plan prioritizes shade trees and covered waiting "
        "areas along "
        "a walking corridor. A field team records canopy cover and surface temperature at"
        " fixed times. "
        "The proposed pilot uses three observation points and repeats measurements after planting. "
        "The brief contains no measured reduction in temperature and makes no claim of "
        "demonstrated effectiveness.",
    ),
    (
        "kano-heat",
        "Northern market shade trial",
        8.5167,
        12.0000,
        "heat",
        "2026-03-16",
        "The fictional northern market trial uses removable shade fabric above a pedestrian queue. "
        "Observers record queue length and shaded pavement temperature. Permanent tree planting is "
        "not proposed because the exercise site has underground service constraints. "
        "Trial results are "
        "not yet available. The planning record must not be interpreted as a real "
        "heat-health advisory.",
    ),
    (
        "lag-transit",
        "Island bus access review",
        3.4200,
        6.4500,
        "transport",
        "2026-04-02",
        "The fictional island bus access review recommends moving the temporary boarding "
        "point away "
        "from a drainage inlet. A marked pedestrian route should connect the stop to the "
        "exercise entrance. "
        "The review records accessibility observations rather than ridership forecasts. "
        "It does not authorize "
        "road construction or changes to a real public transit service.",
    ),
    (
        "abu-transit",
        "Central shuttle scheduling",
        7.4900,
        9.0600,
        "transport",
        "2026-04-03",
        "The fictional central shuttle exercise tests a fifteen-minute departure "
        "interval. Dispatchers "
        "record boarding duration and departure delay at two temporary stops. The "
        "recommendation is to "
        "adjust the timetable only after observing the complete morning session. Moving a"
        " drainage inlet "
        "is outside the trial. No published real-world timetable is represented by this record.",
    ),
    (
        "lag-coast",
        "Coastal erosion survey plan",
        3.4600,
        6.4200,
        "coast",
        "2026-08-01",
        "The fictional coastal survey plan calls for repeat shoreline photographs from "
        "fixed observation "
        "stations. Surveyors must record tidal stage and camera direction with each "
        "image. The brief "
        "proposes comparison of shoreline positions after the exercise but reports no "
        "erosion rate. "
        "Hard coastal defenses are not recommended without a separate engineering and "
        "environmental assessment.",
    ),
    (
        "ph-wetland",
        "Wetland vegetation mapping",
        7.0200,
        4.8000,
        "coast",
        "2026-08-02",
        "The fictional wetland mapping exercise separates mangrove canopy from open water using "
        "field photographs and a small labeled image set. Analysts should check cloud "
        "shadows before "
        "accepting a water classification. The brief proposes manual review of uncertain "
        "pixels and "
        "reports no model accuracy. The location identifies an exercise area, not a "
        "surveyed habitat boundary.",
    ),
    (
        "ibadan-road",
        "Road crossing inspection",
        3.9470,
        7.3775,
        "transport",
        "2026-04-04",
        "The fictional road crossing inspection recommends repainting pedestrian markings"
        " and checking "
        "the visibility of the exercise sign. Observers log conflicts between turning vehicles and "
        "walking participants. No traffic accident statistics were collected. The "
        "recommendation is limited "
        "to an exercise layout and is not a road-safety certification.",
    ),
    (
        "benin-waste",
        "Waste collection routing",
        5.6037,
        6.3350,
        "waste",
        "2026-02-11",
        "The fictional waste collection exercise compares two routes to a temporary "
        "transfer point. "
        "Dispatchers should record travel time and container fill level at each stop. No "
        "optimized route "
        "has been selected yet. The brief asks analysts to keep raw observations so that "
        "routing decisions "
        "can be checked against measured travel times rather than assumed road speeds.",
    ),
    (
        "jos-soil",
        "Slope vegetation inspection",
        8.8583,
        9.8965,
        "land",
        "2026-02-12",
        "The fictional slope inspection documents bare soil patches beside a training footpath. "
        "The proposed response is to restore vegetation and divert exercise foot traffic until the "
        "surface is stable. A geotechnical assessment would be required for any "
        "structural intervention. "
        "This record provides no landslide probability and should not be used for real "
        "slope-risk decisions.",
    ),
    (
        "date-east",
        "Eastern Pacific monitoring buoy",
        179.8,
        0.2,
        "ocean",
        "2026-01-05",
        "The fictional eastern Pacific buoy stores sea-surface temperature observations for an "
        "instrumentation exercise. The operator checks battery voltage before downloading the log. "
        "The anchor lies east of the international date line and is included to test "
        "wrapped geographic "
        "queries. No actual buoy or ocean measurement is represented.",
    ),
    (
        "date-west",
        "Western Pacific monitoring buoy",
        -179.8,
        -0.2,
        "ocean",
        "2026-01-06",
        "The fictional western Pacific buoy stores salinity observations for an instrumentation "
        "exercise. The operator cleans the conductivity probe before the calibration "
        "check. The anchor "
        "lies west of the international date line and must be retrieved by a bounding box"
        " crossing that "
        "meridian. No actual salinity reading is included in the fixture.",
    ),
]


def main():
    features = []
    for identifier, title, lon, lat, topic, published, text in RECORDS:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": identifier,
                    "title": title,
                    "text": text,
                    "topic": topic,
                    "published": published,
                    "source": f"fixture://planning-briefs/{identifier}",
                },
            }
        )
    path = Path(__file__).with_name("demo.geojson")
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
