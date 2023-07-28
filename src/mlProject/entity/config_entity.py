from dataclasses import dataclass
from pathlib import Path

# Definir una clase de datos inmutable que almacena 
# la configuración para la ingesta de datos
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# ------------------------------------------------------------------------------------------------
