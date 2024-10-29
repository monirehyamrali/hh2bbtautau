# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    set_ak_column, remove_ak_column, attach_behavior, EMPTY_FLOAT, get_ak_routes)
from columnflow.types import Sequence
import numpy as np

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags", "GenPart.status"},
    produces={"gen_top_decay.*.pt", "gen_top_decay.*.eta", "gen_top_decay.*.phi", "gen_top_decay.*.mass", "gen_top_decay.*.pdgId",
    "gen_top_decay.*.statusFlags", "z_kaon_pos", "z_kaon_neg", 
    "pion_neg.*", "pion_pos.*",
    "tau_nus.*",
    },
)
def gen_top_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """
    # find hard top quarks
    abs_id = abs(events.GenPart.pdgId)
    top = events.GenPart[abs_id == 6]
    top = top[top.hasFlags("isHardProcess")]
    top = top[~ak.is_none(top, axis=1)]


    t = events.GenPart[events.GenPart.pdgId == 6]
    t = t[t.hasFlags("isHardProcess")]
    t = t[~ak.is_none(t, axis=1)]
    
    t_bar = events.GenPart[events.GenPart.pdgId == -6]
    t_bar = t_bar[t_bar.hasFlags("isHardProcess")]
    t_bar = t_bar[~ak.is_none(t_bar, axis=1)]

    t_children = t.distinctChildren
    abs_t_children_id = abs(t_children.pdgId)
    t_children = ak.drop_none(t_children, behavior=t_children.behavior)
    
    t_bar_children = t_bar.distinctChildren
    abs_t_bar_children_id = abs(t_bar_children.pdgId)
    t_bar_children = ak.drop_none(t_bar_children, behavior=t_bar_children.behavior)

    #from IPython import embed; embed()
    # distinct top quark children (b's and W's)
    top_children = top.distinctChildrenDeep[top.distinctChildrenDeep.hasFlags("isHardProcess")]
    t_children = t.distinctChildrenDeep[t.distinctChildrenDeep.hasFlags("isHardProcess")]
    t_bar_children = t_bar.distinctChildrenDeep[t_bar.distinctChildrenDeep.hasFlags("isHardProcess")]
    # get b's
    #from IPython import embed; embed()
    b = top_children[top_children.pdgId == 5][:, 0]
    b_bar = top_children[top_children.pdgId == -5][:, 1]
    # get W's
    w = top_children[abs(top_children.pdgId) == 24][:, :, 0]
    # w plus
    w_t = t_children[abs(t_children.pdgId) == 24][:, :, 0]
    # w minus
    w_t_bar = t_bar_children[abs(t_bar_children.pdgId) == 24][:, :, 0]

    # distinct W children
    w_children = w.distinctChildrenDeep[w.distinctChildrenDeep.hasFlags("isHardProcess")]

    w_t_children = w_t.distinctChildrenDeep
    w_t_bar_children = w_t_bar.distinctChildrenDeep
   
    # quarks from the w decay 
    q_t = w_t_children[abs(w_t_children.pdgId) < 11]
    q_t_bar = w_t_bar_children[abs(w_t_bar_children.pdgId) < 11]
    # leptons from the w decay
    lep_t = w_t_children[abs(w_t_children.pdgId) >= 11 ]
    lep_t_bar = w_t_bar_children[abs(w_t_bar_children.pdgId) >= 11 ]

    # get tau lepton from w 
    tau_neg = w_t_bar_children[w_t_bar_children.pdgId == 15]
    tau_neg = tau_neg[tau_neg.hasFlags("fromHardProcess")]
    tau_neg = ak.drop_none(tau_neg, behavior=tau_neg.behavior)

    tau_pos = w_t_children[w_t_children.pdgId == -15]
    tau_pos = tau_pos[tau_pos.hasFlags("fromHardProcess")]
    tau_pos = ak.drop_none(tau_pos, behavior=tau_pos.behavior)
    
    
    # distinct children tau
    tau_children_pos = tau_pos.distinctChildren
    abs_tau_children_pos_id = abs(tau_children_pos.pdgId)
    tau_children_pos = ak.drop_none(tau_children_pos, behavior=tau_children_pos.behavior)
    
    tau_children_neg = tau_neg.distinctChildrenDeep
    abs_tau_children_neg_id = abs(tau_children_neg.pdgId)
    tau_children_neg = ak.drop_none(tau_children_neg, behavior=tau_children_neg.behavior)
    #from IPython import embed; embed()
    # tau decays only in one pion/ kaon and neutrino
    tau_neg_2c = tau_children_neg[ak.num(tau_children_neg, axis=3) == 2]
    pion_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -211], axis=2))
    kaon_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -321], axis=2))

    tau_pos_2c = tau_children_pos[ak.num(tau_children_pos, axis=3) == 2]
    pion_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 211], axis=2))
    kaon_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 321], axis=2))

    # the energy fraction of a single pion and the decayed tau
    z_pion_neg = ak.firsts(pion_neg.E)/tau_neg[:, 0].E
    z_pion_pos = ak.firsts(pion_pos.E)/tau_pos[:, 0].E
    z_kaon_bar_neg = ak.firsts(kaon_neg.E)/tau_neg[:, 0].E
    z_kaon_t_pos = ak.firsts(kaon_pos.E)/tau_pos[:, 0].E
   
    # # reorder the first two W children (leptons or quarks) so that the charged lepton / down-type
    # # quark is listed first (they have an odd pdgId)
    # w_children_firsttwo = w_children[:, :, :2]
    # w_children_firsttwo = w_children_firsttwo[(w_children_firsttwo.pdgId % 2 == 0) * 1]
    # w_children_rest = w_children[:, :, 2:]
    # #from IPython import embed; embed()

    # 3 vector of neutrinos from tau decay
    abs_id_neg = abs(tau_neg.distinctChildren.pdgId)
    nu_neg = tau_neg.distinctChildren[(abs_id_neg == 12) | (abs_id_neg == 14) | (abs_id_neg == 16)]
    nus_neg = ak.flatten(nu_neg, axis=1)
    # if there are no taus, only 1-dimensional
    # so, add a new dimensions by padding and filling
    padded_nus_neg = ak.fill_none(ak.pad_none(nus_neg, 1, axis=1), [], axis=1)
    # then, slice to reduce the dimension and get only the neutrinos
    padded_nus_neg_reduced = padded_nus_neg[:,0]
    nu_3_neg = padded_nus_neg_reduced.pvec
    sum_nu_neg = nu_3_neg.sum(axis=-1)

    abs_id_pos = abs(tau_pos.distinctChildren.pdgId)
    nu_pos = tau_pos.distinctChildren[(abs_id_pos == 12) | (abs_id_pos == 14) | (abs_id_pos == 16)]
    nus_pos = ak.flatten(nu_pos, axis=1)

    # if there are no taus, only 1-dimensional
    # so, add a new dimensions by padding and filling
    padded_nus_pos = ak.fill_none(ak.pad_none(nus_pos, 1, axis=1), [], axis=1)
    # then, slice to reduce the dimension and get only the neutrinos
    padded_nus_pos_reduced = padded_nus_pos[:,0]
    nu_3_pos = padded_nus_pos_reduced.pvec

    sum_nu_pos = nu_3_pos.sum(axis=-1)

    print("gen top decay debug console")
    #from IPython import embed; embed()
    tau_nu = ak.concatenate([sum_nu_neg[..., None], sum_nu_pos[..., None]], axis=-1)
    
    






    # # [:, :, None]
    # #concatenate to create the structure to return
    # groups = ak.concatenate(
    #     [
    #         t,
    #         t_bar,
    #         b,
    #         b_bar,
    #         w_t,
    #         w_t_bar,
    #         w_t_children,
    #         w_t_bar_children,
    #         q_t,
    #         q_t_bar,
    #         lep_t,
    #         lep_t_bar,
    #         tau_neg,
    #         tau_pos,
            
    #     ],
    #     axis=1,
    # )

    field_dict = {
        "t": t,
        "t_bar": t_bar,
        # "b": b,
        # "b_bar": b_bar,
        # "w_t": w_t,
        # "tau_neg": tau_neg,
        # "tau_pos": tau_pos,
        # "w_t_bar": w_t_bar,
        # "w_t_children": w_t_children,
        # "w_t_bar_children": w_t_bar_children,
        # "q_t": q_t,
        # "q_t_bar": q_t_bar,
        # "lep_t": lep_t,
        # "lep_t_bar": lep_t_bar,

    }
    
    
    groups = ak.zip({key: ak.pad_none(val, 2, axis=-1) for key, val in field_dict.items()})
    # # save the column
    #from IPython import embed; embed()
    def make_column_save(src: ak.Array):
        col = ak.pad_none(src, 1, axis=-1)
        col = ak.fill_none(col, EMPTY_FLOAT, axis=-1)
        return col

    # helper to clean problematic columns from array
    def set_ak_column_save(
        events: ak.Array,
        target: str,
        src: ak.Array,
        problematic: Sequence[str] or None = None
    ) -> ak.Array:
        if problematic == None:
            problematic = (
                "childrenIdxG", "distinctChildrenDeepIdxG", "distinctChildrenIdxG",
                "distinctParentIdxG", "genPartIdxMother", "genPartIdxMotherG",
            )
        # from IPython import embed; embed()
        for route in get_ak_routes(src):
            if not route in problematic:
                events = set_ak_column(events, ".".join([target, route.column]), make_column_save(route.apply(src)))
        return events
    
    events = set_ak_column_save(events, "gen_top_decay", groups)
    events = set_ak_column(events, "z_kaon_neg", z_kaon_bar_neg)
    events = set_ak_column(events, "z_kaon_pos", z_kaon_t_pos)
    events = set_ak_column_save(events, "pion_neg", pion_neg)
    events = set_ak_column_save(events, "pion_pos", pion_pos)
    events = set_ak_column(events, "pion_neg.zfrac", make_column_save(z_pion_neg))
    events = set_ak_column(events, "pion_pos.zfrac", make_column_save(z_pion_pos))
    events = set_ak_column_save(events, "tau_nus", tau_nu)

    # from IPython import embed; embed()

    return events


@gen_top_decay_products.skip
def gen_top_decay_products_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
