from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
ocean_climate = pd.read_csv(app_dir / "realistic_ocean_climate_dataset.csv")