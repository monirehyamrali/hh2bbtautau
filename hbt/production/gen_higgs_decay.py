# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    set_ak_column, remove_ak_column, attach_behavior, EMPTY_FLOAT, get_ak_routes, remove_ak_column)
from columnflow.types import Sequence
import numpy as np

ak = maybe_import("awkward")


@producer(
    uses={
        "GenPart.genPartIdxMother", "GenPart.pdgId",
        "GenPart.statusFlags", "GenPart.status",
        "GenPart.eta", "GenPart.pt", "GenPart.phi", "GenPart.mass",
        "Tau.pt", "Tau.eta", "Tau.mass", "Tau.charge", "Tau.decayMode",
    },
    # produces={"gen_higgs_decay.*", "z_pion_neg", "z_pion_pos", "z_kaon_neg", "z_kaon_pos", "pion_neg.*", "pion_pos", "pion_neg_E", "pion_pos_E"},
    produces={# "gen_higgs_decay.tau_neg.*",
        "z_kaon_neg", "z_kaon_pos", "pion_neg.*", "pion_pos.*", "tau_nus.*",
        },
)
def gen_higgs_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_higgs_decay" with one element per hard higgs boson. Each element is
    a GenParticleArray with five or more objects in a distinct order: higgs boson, bottom quark, tau lepton,
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
    n = 19362
    #from IPython import embed; embed(header="anfang gen HH")
    # find hard h boson
    abs_id = abs(events.GenPart.pdgId)
    h = events.GenPart[abs_id == 25]
    h = h[h.hasFlags("isLastCopy")]
    h = ak.drop_none(h, behavior=h.behavior)

    # distinct higgs boson children (b's and tau's)
    h_children = h.distinctChildrenDeep
    abs_children_id = abs(h_children.pdgId)

    # get b's
    b = events.GenPart[abs_id == 5]
    b = b[b.hasFlags("isFirstCopy", "fromHardProcess")]
    # remove optional (first remove nones, then update the type)
    b = ak.drop_none(b, behavior=b.behavior)

    # get tau's
    tau = events.GenPart[abs_id == 15]
    tau = tau[tau.hasFlags("isLastCopy", "fromHardProcess")]
    # remove optional
    tau = ak.drop_none(tau, behavior=tau.behavior)
    
    # distinct tau children
    tau_children = tau.distinctChildren
    abs_tau_children_id = abs(tau_children.pdgId)
    tau_children = ak.drop_none(tau_children, behavior=tau_children.behavior)

    # separate tau-positive, tau-negative
    tau_pos = events.GenPart[events.GenPart.pdgId == -15]
    tau_pos = tau_pos[tau_pos.hasFlags("isLastCopy", "fromHardProcess")]
    tau_pos = ak.drop_none(tau_pos, behavior=tau_pos.behavior)
    
    tau_neg = events.GenPart[events.GenPart.pdgId == 15]
    tau_neg = tau_neg[tau_neg.hasFlags("isLastCopy", "fromHardProcess")]
    tau_neg = ak.drop_none(tau_neg, behavior=tau_neg.behavior)
    
    tau_children_pos = tau_pos.distinctChildren
    abs_tau_children_pos_id = abs(tau_children_pos.pdgId)
    tau_children_pos = ak.drop_none(tau_children_pos, behavior=tau_children_pos.behavior)
    
    tau_children_neg = tau_neg.distinctChildren
    abs_tau_children_neg_id = abs(tau_children_neg.pdgId)
    tau_children_neg = ak.drop_none(tau_children_neg, behavior=tau_children_neg.behavior)

    h_tau = tau.parent.parent
    h_tau = ak.drop_none(h_tau, behavior=h_tau.behavior)
   
    h_b = b.parent
    h_b = ak.drop_none(h_b, behavior=h_b.behavior)
    
    # get nu's
    nu = ak.firsts(tau_children[abs_tau_children_id == 16], axis=2)
    # remove optional
    nu = ak.drop_none(nu, behavior=nu.behavior)
    
    # get w's
    w = ak.firsts(tau_children[abs_tau_children_id == 24], axis=2)
    # remove optional
    w = ak.drop_none(w, behavior=w.behavior)
    #from IPython import embed; embed(header="w_children")
    # decays might also be effective into a variable number of mesons -> add them
    w_children = tau_children[abs_tau_children_id != 16]
    # remove optional
    w_children = ak.drop_none(w_children, behavior=w_children.behavior)
    
    # temporarily here: sum children and make them GenParticles
   #  w_sum = w_children.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_sum = set_ak_column(w_sum, c, getattr(w_sum, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_sum = remove_ak_column(w_sum, c)
   #  tau_sign = tau.pdgId // abs(tau.pdgId)
   #  w_sum = set_ak_column(w_sum, "pdgId", tau.pdgId - 39 * tau_sign)
   #  w_sum = attach_behavior(w_sum, "GenParticle")


   # # visible decay of w
   #  w_visible = tau_children[(abs_tau_children_id != 12) & (abs_tau_children_id != 14) & (abs_tau_children_id != 16)]
   #  # remove optional
   #  w_visible = ak.drop_none(w_visible, behavior=w_visible.behavior)
   #  #create a four vector
   #  w_visible_sum = w_visible.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_visible_sum = set_ak_column(w_visible_sum, c, getattr(w_visible_sum, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_visible_sum = remove_ak_column(w_visible_sum, c)
   #  tau_sign = tau.pdgId // abs(tau.pdgId)
   #  w_visible_sum = set_ak_column(w_visible_sum, "pdgId", tau.pdgId - 39 * tau_sign)
   #  w_visible_sum = attach_behavior(w_visible_sum, "GenParticle")


   #  # visible hadronic decay of w
   #  w_vis_had_ne = tau_children_neg[
   #      ((abs(tau_children_neg.pdgId) < 11) | (abs(tau_children_neg.pdgId) > 16)) &
   #      (tau_children_neg.pdgId != 22) &
   #      ~tau_children_neg.hasFlags("isPrompt")
   #  ]

   #  w_vis_had_neg = w_vis_had_ne.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_vis_had_neg = set_ak_column(w_vis_had_neg, c, getattr(w_vis_had_neg, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_vis_had_neg = remove_ak_column(w_vis_had_neg, c)
   #  tau_neg_sign = tau_neg.pdgId // abs(tau_neg.pdgId)
   #  w_vis_had_neg = set_ak_column(w_vis_had_neg, "pdgId", tau_neg.pdgId - 39 * tau_neg_sign)
   #  w_vis_had_neg = attach_behavior(w_vis_had_neg, "GenParticle")


   #  w_vis_had_po = tau_children_pos[
   #      ((abs(tau_children_pos.pdgId) < 11) | (abs(tau_children_pos.pdgId) > 16)) &
   #      (tau_children_pos.pdgId != 22) &
   #      ~tau_children_pos.hasFlags("isPrompt")
   #  ]
   #  w_vis_had_pos = w_vis_had_po.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_vis_had_pos = set_ak_column(w_vis_had_pos, c, getattr(w_vis_had_pos, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_vis_had_pos = remove_ak_column(w_vis_had_pos, c)
   #  tau_pos_sign = tau_pos.pdgId // abs(tau_pos.pdgId)
   #  w_vis_had_pos = set_ak_column(w_vis_had_pos, "pdgId", tau_pos.pdgId - 39 * tau_pos_sign)
   #  w_vis_had_pos = attach_behavior(w_vis_had_pos, "GenParticle")

   #  # ------------------------------
   #  # visible leptonic w decay
   #  w_vis_lep_po = tau_children_pos[
   #      ((abs(tau_children_pos.pdgId) == 11) | (abs(tau_children_pos.pdgId) == 13) | (abs(tau_children_pos.pdgId) == 15)) &
   #      ~tau_children_pos.hasFlags("isPrompt")
   #  ]
    
   #  w_vis_lep_pos = w_vis_lep_po.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_vis_lep_pos = set_ak_column(w_vis_lep_pos, c, getattr(w_vis_lep_pos, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_vis_lep_pos = remove_ak_column(w_vis_lep_pos, c)
   #  w_vis_lep_pos = set_ak_column(w_vis_lep_pos, "pdgId", tau_pos.pdgId - 39 * tau_pos_sign)
   #  w_vis_lep_pos = attach_behavior(w_vis_lep_pos, "GenParticle")


   #  w_vis_lep_ne = tau_children_neg[
   #      ((abs(tau_children_neg.pdgId) == 11) | (abs(tau_children_neg.pdgId) == 13) | (abs(tau_children_neg.pdgId) == 15)) &
   #      ~tau_children_neg.hasFlags("isPrompt")
   #  ]
    
   #  w_vis_lep_neg = w_vis_lep_ne.sum(axis=-1)
   #  for c in ["pt", "eta", "phi", "mass"]:
   #      w_vis_lep_neg = set_ak_column(w_vis_lep_neg, c, getattr(w_vis_lep_neg, c))
   #  for c in ["t", "x", "y", "z"]:
   #      w_vis_lep_neg = remove_ak_column(w_vis_lep_neg, c)
   #  w_vis_lep_neg = set_ak_column(w_vis_lep_neg, "pdgId", tau_neg.pdgId - 39 * tau_neg_sign)
   #  w_vis_lep_neg = attach_behavior(w_vis_lep_neg, "GenParticle")
   #  w_vis_lep_neg
    
    # pion plus from tau neg
    pi_pos_neg = tau_children_neg[(tau_children_neg.pdgId == 211) & ~tau_children_neg.hasFlags("isPrompt")] 
    pion_pos_neg = pi_pos_neg.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_pos_neg = set_ak_column(pion_pos_neg, c, getattr(pion_pos_neg, c))
    for c in ["t", "x", "y", "z"]:
        pion_pos_neg = remove_ak_column(pion_pos_neg, c)
    pion_pos_neg = set_ak_column(pion_pos_neg, "pdgId", 211)
    pion_pos_neg = attach_behavior(pion_pos_neg, "GenParticle")
    # pion plus from tau plus
    pi_pos_pos = tau_children_pos[(tau_children_pos.pdgId == 211) & ~tau_children_pos.hasFlags("isPrompt")] 
    pion_pos_pos = pi_pos_pos.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_pos_pos = set_ak_column(pion_pos_pos, c, getattr(pion_pos_pos, c))
    for c in ["t", "x", "y", "z"]:
        pion_pos_pos = remove_ak_column(pion_pos_pos, c)
    pion_pos_pos = set_ak_column(pion_pos_pos, "pdgId", 211)
    pion_pos_pos = attach_behavior(pion_pos_pos, "GenParticle")
    
    # pion minus from tau minus
    pi_neg_neg = tau_children_neg[(tau_children_neg.pdgId == -211) & ~tau_children_neg.hasFlags("isPrompt")] 
    pion_neg_neg = pi_neg_neg.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_neg_neg = set_ak_column(pion_neg_neg, c, getattr(pion_neg_neg, c))
    for c in ["t", "x", "y", "z"]:
        pion_neg_neg = remove_ak_column(pion_neg_neg, c)
    pion_neg_neg = set_ak_column(pion_neg_neg, "pdgId", -211)
    pion_neg_neg = attach_behavior(pion_neg_neg, "GenParticle")
    # pion minus from tau plus
    pi_neg_pos = tau_children_pos[(tau_children_pos.pdgId == -211) & ~tau_children_pos.hasFlags("isPrompt")] 
    pion_neg_pos = pi_neg_pos.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_neg_pos = set_ak_column(pion_neg_pos, c, getattr(pion_neg_pos, c))
    for c in ["t", "x", "y", "z"]:
        pion_neg_pos = remove_ak_column(pion_neg_pos, c)
    pion_neg_pos = set_ak_column(pion_neg_pos, "pdgId", -211)
    pion_neg_pos = attach_behavior(pion_neg_pos, "GenParticle")
    
    # pion zero
    pion = tau_children_pos[(tau_children_pos.pdgId == 111) & ~tau_children_pos.hasFlags("isPrompt")]
    pion_zero_pos = pion.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_zero_pos = set_ak_column(pion_zero_pos, c, getattr(pion_zero_pos, c))
    for c in ["t", "x", "y", "z"]:
        pion_zero_pos = remove_ak_column(pion_zero_pos, c)
    pion_zero_pos = set_ak_column(pion_zero_pos, "pdgId", 111)
    pion_zero_pos = attach_behavior(pion_zero_pos, "GenParticle")

    pion1 = tau_children_neg[(tau_children_neg.pdgId == 111) & ~tau_children_neg.hasFlags("isPrompt")]
    pion_zero_neg = pion1.sum(axis=-1)
    for c in ["pt", "eta", "phi", "mass"]:
        pion_zero_neg = set_ak_column(pion_zero_neg, c, getattr(pion_zero_neg, c))
    for c in ["t", "x", "y", "z"]:
        pion_zero_neg = remove_ak_column(pion_zero_neg, c)
    pion_zero_neg = set_ak_column(pion_zero_neg, "pdgId", 111)
    pion_zero_neg = attach_behavior(pion_zero_neg, "GenParticle")
    

    # tau decays only in one pion/ kaon and neutrino
    tau_neg_2c = tau_children_neg[ak.num(tau_children_neg, axis=2) == 2]
    pion_neg = ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -211], axis=2)
    kaon_neg = ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -321], axis=2)

    tau_pos_2c = tau_children_pos[ak.num(tau_children_pos, axis=2) == 2]
    pion_pos = ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 211], axis=2)
    kaon_pos = ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 321], axis=2)
    # the energy fraction of a single pion and the decayed tau
    z_pion_neg = pion_neg.E/tau_neg[:, 0].E
    z_pion_pos = pion_pos.E/tau_pos[:, 0].E
    z_kaon_neg = kaon_neg.E/tau_neg[:, 0].E
    z_kaon_pos = kaon_pos.E/tau_pos[:, 0].E

    pion_neg_energy = pion_neg.E
    pion_pos_energy = pion_pos.E



    # 3 vector of neutrinos from tau decay
    abs_id_neg = abs(tau_neg.distinctChildren.pdgId)
    nus_neg = tau_neg.distinctChildren[(abs_id_neg == 12) | (abs_id_neg == 14) | (abs_id_neg == 16)]
    nu_3_neg = nus_neg.pvec[:, 0]
    sum_nu_neg = nu_3_neg.sum(axis=-1)

    abs_id_pos = abs(tau_pos.distinctChildren.pdgId)
    nus_pos = tau_pos.distinctChildren[(abs_id_pos == 12) | (abs_id_pos == 14) | (abs_id_pos == 16)]
    nu_3_pos = nus_pos.pvec[:, 0]
    sum_nu_pos = nu_3_pos.sum(axis=-1)
    tau_nu = ak.concatenate([sum_nu_neg[..., None], sum_nu_pos[..., None]], axis=-1)

    #from IPython import embed; embed()






    def remove_fields(x):
        for f in list(x.fields):
            if f not in tau_neg.fields:
                x = ak.without_field(x, f)
        return x

    # def remove_field(x):
    #     for f in list(x.fields):
    #         if f not in w_visible_sum.fields:
    #             x = ak.without_field(x, f)
    #     return x

    # def remove_had_pos(x):
    #     for f in list(x.fields):
    #         if f not in w_vis_had_pos.fields:
    #             x = ak.without_field(x, f)
    #     return x

    # def remove_had_neg(x):
    #     for f in list(x.fields):
    #         if f not in w_vis_had_neg.fields:
    #             x = ak.without_field(x, f)
    #     return x

    # def remove_lep_pos(x):
    #     for f in list(x.fields):
    #         if f not in w_vis_lep_pos.fields:
    #             x = ak.without_field(x, f)
    #     return x

    # def remove_lep_neg(x):
    #     for f in list(x.fields):
    #         if f not in w_vis_lep_neg.fields:
    #             x = ak.without_field(x, f)
    #     return x


    # concatenate to create the structure to return
    field_dict = {
        # "higgs": remove_fields(h),
        #"h_tau": remove_fields(h_tau),
        #"h_b": remove_fields(h_b),
        # "b": remove_fields(b),
        # "tau": remove_fields(tau),
        "tau_neg": remove_fields(tau_neg),
        # "tau_pos": remove_fields(tau_pos),
        # "nu": remove_fields(nu),
        # "w_sum": ak.concatenate([remove_fields(w), w_sum], axis=1),
        # "w_visible_sum": ak.concatenate([remove_field(w), w_visible_sum], axis=1),
        # "w_vis_had_neg": ak.concatenate([remove_had_neg(w), w_vis_had_neg], axis=1),
        # "w_vis_had_pos": ak.concatenate([remove_had_pos(w), w_vis_had_pos], axis=1),
        # "w_vis_lep_neg": ak.concatenate([remove_lep_neg(w), w_vis_lep_neg], axis=1),
        # "w_vis_lep_pos": ak.concatenate([remove_lep_pos(w), w_vis_lep_pos], axis=1),
        # "pion_neg_neg": remove_fields(pion_neg_neg),
        # "pion_pos_pos": remove_fields(pion_pos_pos),
        # "pion_neg_pos": remove_fields(pion_neg_pos),
        # "pion_pos_neg": remove_fields(pion_pos_neg),
        # "pion_zero_neg": remove_fields(pion_zero_neg),
        # "pion_zero_pos": remove_fields(pion_zero_pos),

    }
    #from IPython import embed; embed()
    groups = ak.zip({key: ak.pad_none(val, 2, axis=-1) for key, val in field_dict.items()})

    # save the column

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
        for route in get_ak_routes(src):
            if not route in problematic:
                events = set_ak_column(events, ".".join([target, route.column]), make_column_save(route.apply(src)))

        return events


    # events = set_ak_column(events, "gen_higgs_decay", groups)
    # events = set_ak_column(events, "z_pion_neg", z_pion_neg)
    # events = set_ak_column(events, "z_pion_pos", z_pion_pos)
    events = set_ak_column(events, "z_kaon_pos", z_kaon_pos)
    events = set_ak_column(events, "z_kaon_neg", z_kaon_neg)
    events = set_ak_column_save(events, "pion_neg", pion_neg)
    events = set_ak_column_save(events, "pion_pos", pion_pos)
    events = set_ak_column(events, "pion_neg.zfrac", make_column_save(z_pion_neg))
    events = set_ak_column(events, "pion_pos.zfrac", make_column_save(z_pion_pos))
    events = set_ak_column(events, "pion_neg_E", pion_neg_energy)
    events = set_ak_column(events, "pion_pos_E", pion_pos_energy)    
    events = set_ak_column(events, "tau_nus", tau_nu)
    return events


# @gen_top_decay_products.skip
# def gen_top_decay_products_skip(self: Producer) -> bool:
#     """
#     Custom skip function that checks whether the dataset is a MC simulation containing top
#     quarks in the first place.
#     """
#     # never skip when there is not dataset
#     if not getattr(self, "dataset_inst", None):
#         return False

#     return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
