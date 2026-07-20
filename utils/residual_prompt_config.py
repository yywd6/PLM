"""Flatten the single maintained NCRP-K1 YAML section."""


def flatten_residual_prompt_config(config):
    config = dict(config)
    residual = config.pop("residual_prompt", None)
    if residual is None:
        return config
    if not isinstance(residual, dict):
        raise ValueError("residual_prompt must be a mapping")
    config.update(
        {
            "residual_prompt_enabled": bool(residual.get("enabled", False)),
            "residual_num_bases": int(residual.get("num_bases", 1)),
            "residual_gamma": float(residual.get("gamma", 1.0)),
            "residual_eps": float(residual.get("eps", 1e-6)),
        }
    )
    return config
