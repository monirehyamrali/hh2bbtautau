# coding: utf-8

"""
Producers that determine the generator-level particles related to a z boson decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    set_ak_column, remove_ak_column, attach_behavior, EMPTY_FLOAT, get_ak_routes, remove_ak_column)
from columnflow.types import Sequence
import numpy as np

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags", "GenPart.status"},
    produces={"gen_z_decay.*.pt", "gen_z_decay.*.eta", "gen_z_decay.*.phi", "gen_z_decay.*.mass", "gen_z_decay.*.pdgId",
    "gen_z_decay.*.statusFlags",
    #"z_kaon_neg", "z_kaon_pos",
    "pion_neg.*", "pion_pos.*", "tau_nus.*",},
)
def gen_z_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_z_decay" with one element per hard z boson. Each element is
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
    
    # find hard z boson
    # abs_id = abs(events.GenPart.pdgId)
    # z = events.GenPart[abs_id == 23]
    # z = z[z.hasFlags("isLastCopy")]
    # z = ak.drop_none(z, behavior=z.behavior)
    


    # distinct z boson children  tau's elektrons muons
    # z_children = z.distinctChildren[z.distinctChildren.hasFlags("isHardProcess")]
    # abs_children_id = abs(z_children.pdgId)

    # get tau's
    # tau = z_children[abs_children_id == 15]
    # tau = tau[tau.hasFlags("isLastCopy", "fromHardProcess")]
    # remove optional
    # tau = ak.drop_none(tau, behavior=tau.behavior)

    # new, fixed tau definition
    # tau = events.GenPart[abs(events.GenPart.pdgId) == 15]
    # tau = tau[tau.hasFlags("isLastCopy", "fromHardProcess")]

    # distinct tau children
    # tau_children = tau.distinctChildren
    # abs_tau_children_id = abs(tau_children.pdgId)
    # tau_children = ak.drop_none(tau_children, behavior=tau_children.behavior)

    # separate tau-positive, tau-negative
    tau_pos = events.GenPart[events.GenPart.pdgId == -15]
    tau_pos = tau_pos[tau_pos.hasFlags("isLastCopy", "fromHardProcess")]
    tau_pos = ak.drop_none(tau_pos, behavior=tau_pos.behavior)
    
    tau_neg = events.GenPart[events.GenPart.pdgId == 15]
    tau_neg = tau_neg[tau_neg.hasFlags("isLastCopy", "fromHardProcess")]
    tau_neg = ak.drop_none(tau_neg, behavior=tau_neg.behavior)
    # the decay products of tau
    tau_children_pos = tau_pos.distinctChildren
    abs_tau_children_pos_id = abs(tau_children_pos.pdgId)
    tau_children_pos = ak.drop_none(tau_children_pos, behavior=tau_children_pos.behavior)
    
    tau_children_neg = tau_neg.distinctChildren
    abs_tau_children_neg_id = abs(tau_children_neg.pdgId)
    tau_children_neg = ak.drop_none(tau_children_neg, behavior=tau_children_neg.behavior)

    #from IPython import embed; embed()
    # tau decays only in one pion/ kaon and neutrino
    tau_neg_2c = tau_children_neg[ak.num(tau_children_neg, axis=2) == 2]
    pion_neg = ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -211], axis=2)
    kaon_neg = ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -321], axis=2)

    tau_pos_2c = tau_children_pos[ak.num(tau_children_pos, axis=2) == 2]
    pion_pos = ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 211], axis=2)
    kaon_pos = ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 321], axis=2)

    # the energy fraction of a single pion and the decayed tau
    z_pion_ne = pion_neg.E/ak.firsts(tau_neg.E)
    z_pi_neg = ak.where(np.isnan(z_pion_ne),EMPTY_FLOAT,z_pion_ne)

    z_pion_po = pion_pos.E/ak.firsts(tau_pos.E)
    z_pi_pos = ak.where(np.isnan(z_pion_po),EMPTY_FLOAT,z_pion_po)

    z_kaon_ne = kaon_neg.E/ak.firsts(tau_pos.E)
    z_kaon_neg = ak.where(np.isnan(z_kaon_ne),EMPTY_FLOAT,z_kaon_ne)

    z_kaon_po = kaon_pos.E/ak.firsts(tau_pos.E)
    z_kaon_pos = ak.where(np.isnan(z_kaon_po),EMPTY_FLOAT,z_kaon_po)


    z_pion_neg = ak.unflatten(ak.firsts(z_pi_neg), 1)
    z_pion_pos = ak.unflatten(ak.firsts(z_pi_pos), 1)
    

    #from IPython import embed; embed()
    # 3 vector of neutrinos from tau decay
    abs_id_neg = abs(tau_neg.distinctChildren.pdgId)
    nus_neg = tau_neg.distinctChildren[(abs_id_neg == 12) | (abs_id_neg == 14) | (abs_id_neg == 16)]

    # if there are no taus, only 1-dimensional
    # so, add a new dimensions by padding and filling
    padded_nus_neg = ak.fill_none(ak.pad_none(nus_neg, 1, axis=1), [], axis=1)
    # then, slice to reduce the dimension and get only the neutrinos
    padded_nus_neg_reduced = padded_nus_neg[:,0]
    nu_3_neg = padded_nus_neg_reduced.pvec
    sum_nu_neg = nu_3_neg.sum(axis=-1)

    abs_id_pos = abs(tau_pos.distinctChildren.pdgId)
    nus_pos = tau_pos.distinctChildren[(abs_id_pos == 12) | (abs_id_pos == 14) | (abs_id_pos == 16)]


    # if there are no taus, only 1-dimensional
    # so, add a new dimensions by padding and filling
    padded_nus_pos = ak.fill_none(ak.pad_none(nus_pos, 1, axis=1), [], axis=1)
    # then, slice to reduce the dimension and get only the neutrinos
    padded_nus_pos_reduced = padded_nus_pos[:,0]
    nu_3_pos = padded_nus_pos_reduced.pvec

    sum_nu_pos = nu_3_pos.sum(axis=-1)
    #from IPython import embed; embed()

    tau_nu = ak.concatenate([sum_nu_neg[..., None], sum_nu_pos[..., None]], axis=-1)
    



    #from IPython import embed; embed()

    
    # concatenate to create the structure to return
    field_dict = {
       "tau_neg": tau_neg,
       "tau_pos": tau_pos,


    }
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
    events = set_ak_column(events, "gen_z_decay", groups)
    events = set_ak_column_save(events, "pion_neg", pion_neg)
    events = set_ak_column_save(events, "pion_pos", pion_pos)
    events = set_ak_column(events, "pion_neg.zfrac", make_column_save(z_pion_neg))
    events = set_ak_column(events, "pion_pos.zfrac", make_column_save(z_pion_pos))
    events = set_ak_column(events, "z_kaon_pos", z_kaon_pos)
    events = set_ak_column(events, "z_kaon_neg", z_kaon_neg)
    events = set_ak_column(events, "tau_nus", tau_nu)
    #from IPython import embed; embed()

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
