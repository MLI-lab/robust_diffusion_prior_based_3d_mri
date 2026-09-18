from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from prefect.futures import PrefectFuture

Label = Tuple[str, Any]
Labels = List[Label]
DatasetOutput = Tuple[Optional[Dict[str, Any]], Labels]
LabeledFuture = Tuple[PrefectFuture, Labels]

