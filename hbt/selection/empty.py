# coding: utf-8

"""
Empty selection.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox

from hbt.production.top_spins import top_spins, higgs_spins, z_spins
from hbt.production.tau_zfrac import top_nu, higgs_nu, z_nu

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        mc_weight, pu_weight, process_ids, increment_stats, attach_coffea_behavior, top_spins,top_nu, higgs_nu, z_nu, 
        pdf_weights, murmuf_weights, higgs_spins, z_spins,
    },
    produces={
        mc_weight, pu_weight, process_ids, increment_stats, top_spins, pdf_weights, murmuf_weights,
        higgs_spins, z_spins,top_nu, higgs_nu, z_nu,
    },
    exposed=True,
)
def empty(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # mc-only functions
    if self.dataset_inst.is_mc:
        # add corrected mc weights
        events = self[mc_weight](events, **kwargs)

        # pdf weights
        events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        events = self[murmuf_weights](events, **kwargs)
        
        # pileup weights
        events = self[pu_weight](events, **kwargs)

        # add spin variables
        if self.dataset_inst.name.startswith("tt_"):  # TODO: better switch
            events = self[top_spins](events, **kwargs)
        elif self.dataset_inst.name.startswith("hh_"):  # TODO: better switch
            events = self[higgs_spins](events, **kwargs)
        elif self.dataset_inst.name.startswith("dy_"):  # TODO: better switch
            events = self[z_spins](events, **kwargs)
        if self.dataset_inst.name.startswith("tt_"):  # TODO: better switch
            events = self[top_nu](events, **kwargs)
        elif self.dataset_inst.name.startswith("hh_"):  # TODO: better switch
            events = self[higgs_nu](events, **kwargs)
        elif self.dataset_inst.name.startswith("dy_"):  # TODO: better switch
            events = self[z_nu](events, **kwargs)       

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # combined event selection after all steps
    event_sel = events.event > 0
    results.main["event"] = event_sel

    # create process ids
    events = self[process_ids](events, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": event_sel,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_sel)
        # pu weights with variations
        for name in sorted(self[pu_weight].produces):
            weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)
        # pdf and murmuf weights with variations
        for v in ["", "_up", "_down"]:
            weight_map[f"sum_pdf_weight{v}"] = events[f"pdf_weight{v}"]
            weight_map[f"sum_pdf_weight{v}_selected"] = (events[f"pdf_weight{v}"], event_sel)
            weight_map[f"sum_murmuf_weight{v}"] = events[f"murmuf_weight{v}"]
            weight_map[f"sum_murmuf_weight{v}_selected"] = (events[f"murmuf_weight{v}"], event_sel)
        # groups
        group_map = {
            **group_map,
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results
