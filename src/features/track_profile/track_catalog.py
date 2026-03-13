from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class TrackProfile:
    straight_pct: float                              
    fast_corner_ratio: float                                    
    lap_km: float                                 
    braking_ev_per_lap: int                                    
    drs_zones: int                              
    tdeg_index: float                                          
    aero_df_index: float                                                          
    pit_loss_s: float                                                               
    fuel_effect_s_per10kg: float                              

                                                          
DEFAULT_PROFILE = TrackProfile(
    straight_pct=0.45,
    fast_corner_ratio=0.45,
    lap_km=5.0,
    braking_ev_per_lap=7,
    drs_zones=2,
    tdeg_index=0.55,
    aero_df_index=0.60,
    pit_loss_s=20.0,
    fuel_effect_s_per10kg=0.21,
)

                            
TRACK_ALIASES: Dict[str, str] = {
    "spa": "belgian_grand_prix",
    "monza": "italian_grand_prix",
    "suzuka": "japanese_grand_prix",
    "zandvoort": "dutch_grand_prix",
    "hungaroring": "hungarian_grand_prix",
    "barcelona": "spanish_grand_prix",
    "montreal": "canadian_grand_prix",
    "imola": "emilia_romagna_grand_prix",
    "interlagos": "sao_paulo_grand_prix",
    "austin": "united_states_grand_prix",
    "cota": "united_states_grand_prix",
    "miami": "miami_grand_prix",
    "las_vegas": "las_vegas_grand_prix",
    "lusail": "qatar_grand_prix",
    "yas_marina": "abu_dhabi_grand_prix",
    "shanghai": "chinese_grand_prix",
    "albert_park": "australian_grand_prix",
    "jeddah": "saudi_arabian_grand_prix",
    "baku": "azerbaijan_grand_prix",
    "usa_grand_prix": "united_states_grand_prix",
    "brazilian_grand_prix": "sao_paulo_grand_prix",
    "mexico_city_grand_prix": "mexican_grand_prix",
    "monaco": "monaco_grand_prix",
}

def normalize_track_slug(slug: str) -> str:
    s = (slug or "").strip().lower().replace(" ", "_")
    return TRACK_ALIASES.get(s, s)

TRACK_TO_PROFILE: Dict[str, TrackProfile] = {
    "belgian_grand_prix": TrackProfile(
        straight_pct=0.58, fast_corner_ratio=0.55, lap_km=7.004, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.65, aero_df_index=0.70,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.23,
    ),
    "italian_grand_prix": TrackProfile(
        straight_pct=0.70, fast_corner_ratio=0.30, lap_km=5.793, braking_ev_per_lap=8,
        drs_zones=2, tdeg_index=0.45, aero_df_index=0.30,
        pit_loss_s=18.5, fuel_effect_s_per10kg=0.20,
    ),
    "japanese_grand_prix": TrackProfile(
        straight_pct=0.52, fast_corner_ratio=0.65, lap_km=5.807, braking_ev_per_lap=6,
        drs_zones=1, tdeg_index=0.60, aero_df_index=0.85,
        pit_loss_s=19.5, fuel_effect_s_per10kg=0.22,
    ),
    "singapore_grand_prix": TrackProfile(
        straight_pct=0.18, fast_corner_ratio=0.10, lap_km=4.94, braking_ev_per_lap=9,
        drs_zones=2, tdeg_index=0.62, aero_df_index=0.85,
        pit_loss_s=24.0, fuel_effect_s_per10kg=0.20,
    ),
    "bahrain_grand_prix": TrackProfile(
        straight_pct=0.50, fast_corner_ratio=0.35, lap_km=5.412, braking_ev_per_lap=8,
        drs_zones=3, tdeg_index=0.65, aero_df_index=0.55,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.22,
    ),
    "saudi_arabian_grand_prix": TrackProfile(
        straight_pct=0.60, fast_corner_ratio=0.60, lap_km=6.174, braking_ev_per_lap=6,
        drs_zones=3, tdeg_index=0.35, aero_df_index=0.45,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.20,
    ),
    "australian_grand_prix": TrackProfile(
        straight_pct=0.46, fast_corner_ratio=0.50, lap_km=5.278, braking_ev_per_lap=6,
        drs_zones=4, tdeg_index=0.50, aero_df_index=0.60,
        pit_loss_s=22.0, fuel_effect_s_per10kg=0.21,
    ),
    "chinese_grand_prix": TrackProfile(
        straight_pct=0.45, fast_corner_ratio=0.45, lap_km=5.451, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.55, aero_df_index=0.65,
        pit_loss_s=21.5, fuel_effect_s_per10kg=0.21,
    ),
    "miami_grand_prix": TrackProfile(
        straight_pct=0.48, fast_corner_ratio=0.35, lap_km=5.412, braking_ev_per_lap=7,
        drs_zones=3, tdeg_index=0.50, aero_df_index=0.60,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.21,
    ),
    "emilia_romagna_grand_prix": TrackProfile(
        straight_pct=0.40, fast_corner_ratio=0.50, lap_km=4.909, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.55, aero_df_index=0.75,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.20,
    ),
    "monaco_grand_prix": TrackProfile(
        straight_pct=0.16, fast_corner_ratio=0.10, lap_km=3.337, braking_ev_per_lap=10,
        drs_zones=1, tdeg_index=0.30, aero_df_index=0.95,
        pit_loss_s=18.0, fuel_effect_s_per10kg=0.17,
    ),
    "canadian_grand_prix": TrackProfile(
        straight_pct=0.55, fast_corner_ratio=0.25, lap_km=4.361, braking_ev_per_lap=8,
        drs_zones=3, tdeg_index=0.40, aero_df_index=0.50,
        pit_loss_s=18.5, fuel_effect_s_per10kg=0.19,
    ),
    "spanish_grand_prix": TrackProfile(
        straight_pct=0.42, fast_corner_ratio=0.55, lap_km=4.657, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.65, aero_df_index=0.75,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.21,
    ),
    "austrian_grand_prix": TrackProfile(
        straight_pct=0.60, fast_corner_ratio=0.40, lap_km=4.318, braking_ev_per_lap=7,
        drs_zones=3, tdeg_index=0.50, aero_df_index=0.55,
        pit_loss_s=18.0, fuel_effect_s_per10kg=0.18,
    ),
    "british_grand_prix": TrackProfile(
        straight_pct=0.45, fast_corner_ratio=0.70, lap_km=5.891, braking_ev_per_lap=6,
        drs_zones=2, tdeg_index=0.45, aero_df_index=0.70,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.22,
    ),
    "hungarian_grand_prix": TrackProfile(
        straight_pct=0.24, fast_corner_ratio=0.35, lap_km=4.381, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.60, aero_df_index=0.85,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.20,
    ),
    "dutch_grand_prix": TrackProfile(
        straight_pct=0.30, fast_corner_ratio=0.55, lap_km=4.259, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.55, aero_df_index=0.80,
        pit_loss_s=20.0, fuel_effect_s_per10kg=0.19,
    ),
    "azerbaijan_grand_prix": TrackProfile(
        straight_pct=0.62, fast_corner_ratio=0.25, lap_km=6.003, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.40, aero_df_index=0.45,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.21,
    ),
    "united_states_grand_prix": TrackProfile(
        straight_pct=0.45, fast_corner_ratio=0.50, lap_km=5.513, braking_ev_per_lap=8,
        drs_zones=2, tdeg_index=0.60, aero_df_index=0.70,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.22,
    ),
    "mexican_grand_prix": TrackProfile(
        straight_pct=0.40, fast_corner_ratio=0.35, lap_km=4.304, braking_ev_per_lap=7,
        drs_zones=3, tdeg_index=0.45, aero_df_index=0.75,
        pit_loss_s=18.5, fuel_effect_s_per10kg=0.18,
    ),
    "sao_paulo_grand_prix": TrackProfile(
        straight_pct=0.45, fast_corner_ratio=0.45, lap_km=4.309, braking_ev_per_lap=7,
        drs_zones=2, tdeg_index=0.60, aero_df_index=0.65,
        pit_loss_s=18.5, fuel_effect_s_per10kg=0.19,
    ),
    "las_vegas_grand_prix": TrackProfile(
        straight_pct=0.62, fast_corner_ratio=0.20, lap_km=6.201, braking_ev_per_lap=6,
        drs_zones=2, tdeg_index=0.35, aero_df_index=0.35,
        pit_loss_s=20.0, fuel_effect_s_per10kg=0.20,
    ),
    "qatar_grand_prix": TrackProfile(
        straight_pct=0.40, fast_corner_ratio=0.65, lap_km=5.419, braking_ev_per_lap=6,
        drs_zones=2, tdeg_index=0.70, aero_df_index=0.70,
        pit_loss_s=21.0, fuel_effect_s_per10kg=0.21,
    ),
    "abu_dhabi_grand_prix": TrackProfile(
        straight_pct=0.45, fast_corner_ratio=0.45, lap_km=5.281, braking_ev_per_lap=8,
        drs_zones=3, tdeg_index=0.45, aero_df_index=0.60,
        pit_loss_s=20.0, fuel_effect_s_per10kg=0.20,
    ),

                                                          
    "portuguese_grand_prix": DEFAULT_PROFILE,
    "french_grand_prix": DEFAULT_PROFILE,
    "german_grand_prix": DEFAULT_PROFILE,
    "turkish_grand_prix": DEFAULT_PROFILE,
}
