#%%

import config
from hardware import SLMManager
from phase_generators import generate_optimized_pattern
from visualization import plot_live_update, plot_final_results

def run_once_optimized(debug=False):
    """Run the optimized-microlens pipeline once using defaults."""
    slm = SLMManager(sim_mode=True)

    # Build params from UI defaults
    params = {
        'focal_length_coarse': config.UI_DEFAULTS['focal_length_coarse'],
        'focal_length_fine':   config.UI_DEFAULTS['focal_length_fine'],
        'rows':                config.UI_DEFAULTS['rows'],
        'cols':                config.UI_DEFAULTS['cols'],
        'overlap_ratio':       config.UI_DEFAULTS['overlap_ratio'],
        'dof_factor':          config.UI_DEFAULTS['dof_factor'],
        'size_factor':         config.UI_DEFAULTS['size_factor'],
        'psf_energy_level':    config.UI_DEFAULTS['psf_energy_level'],
        'phase_range':         config.UI_DEFAULTS['phase_range'],
        'lr':                  config.UI_DEFAULTS['lr'],
        'ni':                  config.UI_DEFAULTS['ni'],
        'lens_type':           False,           # Convex (+); keep behavior consistent with UI
        'shape':               slm.shape,       # e.g., (height, width)
    }

    print("== Optimized Microlens: params snapshot ==")
    print({k: params[k] for k in ('focal_length_coarse','focal_length_fine','rows','cols','lr','ni')})

    # If you want to stop BEFORE entering the inner function, set debug=True
    if debug:
        breakpoint()

    # Generate phase and visualize live updates as defined in your project
    phi, optimizer_obj = generate_optimized_pattern(params, plot_live_update)

    # Upload to (simulated) SLM and plot final results
    slm.upload(phi)
    try:
        plot_final_results(optimizer_obj, optimizer_obj.phase_param.detach())
    except Exception:
        # In case your visualization expects a second arg but it's not available
        plot_final_results(optimizer_obj, None)

    print("Run complete.")
    return phi, optimizer_obj, slm

# ---- Execute once (set debug=True to break before calling the inner function) ----
phi, optimizer_obj, slm = run_once_optimized(debug=False)
