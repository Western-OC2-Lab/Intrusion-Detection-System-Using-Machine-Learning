"""Ramo supervisionado — fases 1–6.

Paper/CICIDS: ``data/pipeline_mth_ids_merged``
CAN intrusion: ``--protocol can`` → ``data/pipeline_can_intrusion_merged``
CAN OTIDS: ``--protocol can_otids`` → ``data/pipeline_can_otids_merged``
"""

from __future__ import annotations

from mth_ids_pipeline.orchestration.experiment_runner import build_arg_parser, config_from_args, run_experiment
from mth_ids_pipeline.protocol import get_protocol_settings, is_can_protocol, is_unsw_protocol


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS supervisionado (fases 1–6)"
    )
    parser.set_defaults(from_phase=1, to=6, label_profile="merged")
    args = parser.parse_args()
    if is_can_protocol(args.protocol) or is_unsw_protocol(args.protocol):
        ps = get_protocol_settings(args.protocol)
        if args.label_profile in (None, "merged"):
            args.label_profile = ps.supervised_profile
    run_experiment(config_from_args(args, branch="supervised"))


if __name__ == "__main__":
    main()
