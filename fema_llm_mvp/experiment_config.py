from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    inner_zip: str = "fema_national_household_survey_2023.zip"
    xlsx_name: str = "fema_national_household_survey_2023_data_and_codebook.xlsx"
    sheet_name: str = "Core Survey"
    target: str = "dis_2_prepstages"
    labels: tuple[str, ...] = ("Prepared Individuals", "Unprepared Individuals")
    question_text: str = (
        "Thinking about preparing yourself for a disaster, which of the following "
        "best represents this person's preparedness status?"
    )
    field_labels: dict[str, str] = field(default_factory=lambda: {
        "state": "state or territory",
        "geographic_division": "geographic division",
        "census_region": "census region",
        "rurality": "rurality",
        "age": "age group",
        "sex": "sex",
        "education": "education",
        "race_selfid": "race",
        "ethnicity": "Hispanic/Latino origin",
        "income_agg": "annual household income",
        "dis_perception": "perceived likelihood that a disaster would impact them",
        "dis_exp": "whether respondent/family experienced disaster impacts",
        "dis_stepshelp": "belief that preparation would help",
        "dis_confidence": "confidence in taking preparedness steps",
        "disability": "disability or health condition affecting emergency response",
        "care": "responsibility for assisting an elderly person or someone with disability",
        "numadult": "number of adults in household",
        "numchild": "number of children in household",
        "homeownership": "home tenure",
        "finres_insuranceresidence": "homeowners or renters insurance",
    })
    disclosure_fields: dict[str, list[str]] = field(default_factory=lambda: {
        "C0": [],
        "C1": [
            "age",
            "sex",
            "education",
            "race_selfid",
            "ethnicity",
            "income_agg",
        ],
        "C2": [
            "age",
            "sex",
            "education",
            "race_selfid",
            "ethnicity",
            "income_agg",
            "state",
            "geographic_division",
            "census_region",
            "rurality",
            "dis_perception",
            "dis_exp",
        ],
        "C3": [
            "age",
            "sex",
            "education",
            "race_selfid",
            "ethnicity",
            "income_agg",
            "state",
            "geographic_division",
            "census_region",
            "rurality",
            "dis_perception",
            "dis_exp",
            "dis_stepshelp",
            "dis_confidence",
            "disability",
            "care",
            "numadult",
            "numchild",
            "homeownership",
            "finres_insuranceresidence",
        ],
    })


DEFAULT_EXPERIMENT = ExperimentConfig()

