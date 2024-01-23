# coding: utf-8

"""
Producers that determine the generator-level particles related to a z boson decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, remove_ak_column, attach_behavior, EMPTY_FLOAT
import numpy as np

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags", "GenPart.status"},
    produces={"gen_z_decay.*","z_pion_neg", "z_pion_pos", "z_kaon_neg", "z_kaon_pos", "pion_neg", "pion_pos"},
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
    abs_id = abs(events.GenPart.pdgId)
    z = events.GenPart[abs_id == 23]
    z = z[z.hasFlags("isLastCopy")]
    z = ak.drop_none(z, behavior=z.behavior)
    


    # distinct z boson children  tau's elektrons muons
    z_children = z.distinctChildrenDeep[z.distinctChildrenDeep.hasFlags("isHardProcess")]
    abs_children_id = abs(z_children.pdgId)

    # get tau's
    tau = z_children[abs_children_id == 15]
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



    tau_neg_2c = tau_children_neg[ak.num(tau_children_neg, axis=2) == 2]
    pion_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -211], axis=2))
    kaon_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -321], axis=2))

    tau_pos_2c = tau_children_pos[ak.num(tau_children_pos, axis=2) == 2]
    pion_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 211], axis=2))
    kaon_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 321], axis=2))


    z_pion_ne = pion_neg.E/tau_neg.E
    z_pion_neg = ak.where(np.isnan(z_pion_ne),EMPTY_FLOAT,z_pion_ne)

    z_pion_po = pion_pos.E/tau_pos.E
    z_pion_pos = ak.where(np.isnan(z_pion_po),EMPTY_FLOAT,z_pion_po)

    z_kaon_ne = kaon_neg.E/tau_neg.E
    z_kaon_neg = ak.where(np.isnan(z_kaon_ne),EMPTY_FLOAT,z_kaon_ne)

    z_kaon_po = kaon_pos.E/tau_pos.E
    z_kaon_pos = ak.where(np.isnan(z_kaon_po),EMPTY_FLOAT,z_kaon_po)

   
    
    



    

    
    # concatenate to create the structure to return
    field_dict = {
       "tau_neg": tau_neg,
       "tau_pos": tau_pos,


    }
    groups = ak.zip({key: ak.pad_none(val, 2, axis=-1) for key, val in field_dict.items()})
    
    # save the column
    events = set_ak_column(events, "gen_z_decay", groups)
    events = set_ak_column(events, "z_pion_neg", z_pion_neg)
    events = set_ak_column(events, "z_pion_pos", z_pion_pos)
    events = set_ak_column(events, "z_kaon_pos", z_kaon_pos)
    events = set_ak_column(events, "z_kaon_neg", z_kaon_neg)
    events = set_ak_column(events, "pion_neg", pion_neg)
    events = set_ak_column(events, "pion_pos", pion_pos)

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
