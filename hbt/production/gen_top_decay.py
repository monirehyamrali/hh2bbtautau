# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, remove_ak_column, attach_behavior, EMPTY_FLOAT
import numpy as np

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags", "GenPart.status"},
    produces={"gen_top_decay", "z_kaon_pos", "z_pion_pos", "z_pion_neg", "z_kaon_neg", 
    "pion_neg", "pion_pos"},
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
    b = top_children[abs(top_children.pdgId) == 5][:, :, 0]

    # get W's
    w = top_children[abs(top_children.pdgId) == 24][:, :, 0]
    w_t = t_children[abs(t_children.pdgId) == 24][:, :, 0]
    w_t_bar = t_bar_children[abs(t_bar_children.pdgId) == 24][:, :, 0]

    # distinct W children
    w_children = w.distinctChildrenDeep[w.distinctChildrenDeep.hasFlags("isHardProcess")]

    w_t_children = w_t.distinctChildrenDeep
    w_t_bar_children = w_t_bar.distinctChildrenDeep
    #from IPython import embed; embed()
    q_t = w_t_children[abs(w_t_children.pdgId) < 11]
    q_t_bar = w_t_bar_children[abs(w_t_bar_children.pdgId) < 11]
    lep_t = w_t_children[abs(w_t_children.pdgId) >= 11 ]
    lep_t_bar = w_t_bar_children[abs(w_t_bar_children.pdgId) >= 11 ]


    tau_neg = w_t_bar_children[w_t_bar_children.pdgId == 15]
    tau_neg = tau_neg[tau_neg.hasFlags("isLastCopy", "fromHardProcess")]
    tau_neg = ak.drop_none(tau_neg, behavior=tau_neg.behavior)
    

    tau_pos = w_t_children[w_t_children.pdgId == -15]
    tau_pos = tau_pos[tau_pos.hasFlags("isLastCopy", "fromHardProcess")]
    tau_pos = ak.drop_none(tau_pos, behavior=tau_pos.behavior)

    

    tau_children_pos = tau_pos.distinctChildren
    abs_tau_children_pos_id = abs(tau_children_pos.pdgId)
    tau_children_pos = ak.drop_none(tau_children_pos, behavior=tau_children_pos.behavior)
    
    tau_children_neg = tau_neg.distinctChildrenDeep
    abs_tau_children_neg_id = abs(tau_children_neg.pdgId)
    tau_children_neg = ak.drop_none(tau_children_neg, behavior=tau_children_neg.behavior)


    tau_neg_2c = tau_children_neg[ak.num(tau_children_neg, axis=3) == 2]
    pion_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -211], axis=2))
    kaon_neg = ak.firsts(ak.flatten(tau_neg_2c[tau_neg_2c.pdgId == -321], axis=2))

    tau_pos_2c = tau_children_pos[ak.num(tau_children_pos, axis=3) == 2]
    pion_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 211], axis=2))
    kaon_pos = ak.firsts(ak.flatten(tau_pos_2c[tau_pos_2c.pdgId == 321], axis=2))

    z_pion_bar_neg = pion_neg.E/ak.firsts(tau_neg[:, 0].E)
    z_pion_t_pos = pion_pos.E/ak.firsts(tau_pos[:, 0].E)
    z_kaon_bar_neg = kaon_neg.E/ak.firsts(tau_neg[:, 0].E)
    z_kaon_t_pos = kaon_pos.E/ak.firsts(tau_pos[:, 0].E)
   
    # reorder the first two W children (leptons or quarks) so that the charged lepton / down-type
    # quark is listed first (they have an odd pdgId)
    w_children_firsttwo = w_children[:, :, :2]
    w_children_firsttwo = w_children_firsttwo[(w_children_firsttwo.pdgId % 2 == 0) * 1]
    w_children_rest = w_children[:, :, 2:]

    
    # # with negative charge
    # electron = w_t_bar_children[w_t_bar_children.pdgId == 11]
    # e_num = ak.sum(w_t_bar_children.pdgId == 11)
    # #16734
    # tau_num = ak.sum(w_t_bar_children.pdgId == 15)
    # #16591
    # muon = w_t_bar_children[w_t_bar_children.pdgId == 13]
    # muon_num = ak.sum(w_t_bar_children.pdgId == 13)
    # #16617

    # # with positive charge
    # muon_pos = w_t_children[w_t_children.pdgId == -13]
    # muon_pos_num = ak.sum(w_t_children.pdgId == -13)
    # #16756

    # positron = w_t_children[w_t_children.pdgId == -11]
    # eplus_num = ak.sum(w_t_children.pdgId == -11)
    # #16616
    
    # anti_tau_num = ak.sum(w_t_children.pdgId == -15)
    # #16715

    # quarks = w_t_children[abs(w_t_children.pdgId) < 11]
    # quarks_num = ak.sum(abs(w_t_children.pdgId) < 11)
    # #99860
    
    # antiquarks = w_t_bar_children[abs(w_t_bar_children.pdgId) < 11]
    # antiquarks_num = ak.sum(abs(w_t_bar_children.pdgId) < 11)
    # #100140
    # from IPython import embed; embed()

    # #both ww
    # electron = w_children[w_children.pdgId == 11]
    # e_num = ak.sum(w_children.pdgId == 11)
    # #16722


    # muon = w_children[w_children.pdgId == 13]
    # mu_num = ak.sum(w_children.pdgId == 13)
    # #16617
    
    # tau = w_children[w_children.pdgId == 15]
    # tau_num = ak.sum(w_children.pdgId == 15)
    # #16591

    # antitau = w_children[w_children.pdgId == -15]
    # antitau_num = ak.sum(w_children.pdgId == -15)
    # #16715
    
    # positron = w_children[w_children.pdgId == -11]
    # posi_num = ak.sum(w_children.pdgId == -11)
    # #16599

    # antimuon = w_children[w_children.pdgId == -13]
    # antimu_num = ak.sum(w_children.pdgId == -13)
    # #16756

    # q = w_children[abs(w_children.pdgId) < 11]
    # q_num = ak.sum(abs(w_children.pdgId) < 11)







    # [:, :, None]
    #concatenate to create the structure to return
    groups = ak.concatenate(
        [
            t,
            t_bar,
            b,
            w_t,
            w_t_bar,
            w_t_children,
            w_t_bar_children,
            q_t,
            q_t_bar,
            lep_t,
            lep_t_bar,
            tau_neg,
            tau_pos,
            
        ],
        axis=1,
    )
    #from IPython import embed; embed()

    # field_dict = {
    #    "t": t,
    #    "t_bar": t_bar,
    #    "b": b,
    #    "w_t": w_t,
    #    "w_t_bar": w_t_bar,
    #    "w_t_children": w_t_children,
    #    "w_t_bar_children": w_t_bar_children,
    #    "q_t": q_t,
    #    "q_t_bar": q_t_bar,
    #    "lep_t": lep_t,
    #    "lep_t_bar": lep_t_bar, 

    # }
    
    
    # groups = ak.zip({key: ak.pad_none(val, 2, axis=-1) for key, val in field_dict.items()})
    # # save the column
    events = set_ak_column(events, "gen_top_decay", groups)
    events = set_ak_column(events, "z_pion_neg", z_pion_bar_neg)
    events = set_ak_column(events, "z_pion_pos", z_pion_t_pos)
    events = set_ak_column(events, "z_kaon_neg", z_kaon_bar_neg)
    events = set_ak_column(events, "z_kaon_pos", z_kaon_t_pos)
    # events = set_ak_column(events, "z_pion_det_neg", 
    #     ak.where(ak.is_none(z_pion_det_neg), EMPTY_FLOAT, z_pion_det_neg))
    # events = set_ak_column(events, "z_pion_det_pos", 
    #     ak.where(ak.is_none(z_pion_det_pos), EMPTY_FLOAT, z_pion_det_pos))
    events = set_ak_column(events, "pion_neg", pion_neg)
    events = set_ak_column(events, "pion_pos", pion_pos)


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
