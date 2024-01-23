# coding: utf-8

"""
Top spin variables.
"""

import functools
import itertools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, flat_np_view, ak_copy, EMPTY_FLOAT, optional_column

from hbt.production.gen_top_decay import gen_top_decay_products
from hbt.production.gen_higgs_decay import gen_higgs_decay_products
from hbt.production.gen_z_decay import gen_z_decay_products


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        gen_top_decay_products, 
        "GenPart.eta", "GenPart.pt", "GenPart.phi", "GenPart.mass",
    },
    produces={
        *optional_column("ts_t", "z_pion_det_pos", "z_pion_det_neg"),
        "z_kaon_pos", "z_pion_pos", "z_pion_neg", "z_kaon_neg", "pion_neg", "pion_pos",
    },
    # only run on mc
    mc_only=True,
)
def top_spins(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_top_decay_products](events, **kwargs)
    
    t = events.gen_top_decay[:, 0]
    t_bar = events.gen_top_decay[:, 1]
    b = events.gen_top_decay[:, 2:4]
    w_t = events.gen_top_decay[:, 4]
    w_t_bar = events.gen_top_decay[:, 5]
    w_t_children = events.gen_top_decay[:, 6]
    w_t_bar_children = events.gen_top_decay[:, 7]
    q_t = events.gen_top_decay[:, 8]
    q_t_bar = events.gen_top_decay[:, 9]
    lep_t = events.gen_top_decay[:, 10]
    lep_t_bar = events.gen_top_decay[:, 11]
    tau_neg = events.gen_top_decay[:, 12]
    tau_pos = events.gen_top_decay[:, 13]
    
    # # create the threevector
    # w3 =  w.boost(-t.boostvec).pvec
    # l3 =  l_dq.boost(-w.boostvec).pvec
    # nu3 = nu_uq.boost(-w.boostvec).pvec
    w_t3 =  w_t.boost(-t.boostvec).pvec
    w_t_bar3 =  w_t_bar.boost(-t_bar.boostvec).pvec
    q_t3 = q_t.boost(-w_t.boostvec).pvec
    q_t_bar3 = q_t_bar.boost(-w_t_bar.boostvec).pvec

   
    # calculate the cosine between lepton or up_type quark/ nu or down_type quark and w boson 
    # cos_l_dq = (l3.px * w3.px + l3.py * w3.py + l3.pz * w3.pz ) / (l3.p * w3.p)
    # cos_nu_uq = (nu3.px * w3.px + nu3.py * w3.py + nu3.pz * w3.pz ) / (nu3.p * w3.p)

    # lep_mask = abs(events.gen_top_decay[:, 0, 4].pdgId) >= 12
    # cos_l = np.ones(len(events), dtype=np.float32) * EMPTY_FLOAT
    # cos_nu = np.ones(len(events), dtype=np.float32) * EMPTY_FLOAT
    # cos_l[lep_mask] = cos_l_dq[lep_mask]
    # cos_nu[lep_mask] = cos_nu_uq[lep_mask]

    # cos_u = np.ones(len(events), dtype=np.float32) * EMPTY_FLOAT
    # cos_d = np.ones(len(events), dtype=np.float32) * EMPTY_FLOAT
    # cos_u[~lep_mask] = cos_nu_uq[~lep_mask]
    # cos_d[~lep_mask] = cos_l_dq[~lep_mask]


    # w0_mass = events.gen_top_decay[:, 0, 2].mass
    # z_pion_det_neg = pion_det_neg.E/ak.firsts(tau_neg[:, 0].E)
    # z_pion_det_pos = pion_det_pos.E/ak.firsts(tau_pos[:, 0].E)

    
    # events = set_ak_column_f32(events, "ts_cos_l", cos_l)
    # events = set_ak_column_f32(events, "ts_cos_nu", cos_nu)
    # events = set_ak_column_f32(events, "ts_cos_u", cos_u)
    # events = set_ak_column_f32(events, "ts_cos_d", cos_d)
    events = set_ak_column_f32(events, "ts_t", t)
    

    return events


@producer(
    uses={
        gen_higgs_decay_products,
        "GenPart.eta", "GenPart.pt", "GenPart.phi", "GenPart.mass", 
    },
    produces={
        # gen_higgs_decay_products,
        *optional_column("ts_cos_bb", "ts_cos_tautau",
         "ts_cos_hh", "ts_cos_tau_neg", "ts_cos_tau_pos", 
         "ts_z_had_pos", "ts_z_had_neg", "ts_z_lep_pos",
         "ts_z_lep_neg", 
         "ts_z_anti_nu", "ts_z_nu_tau"), 
         # "z_pion_pos", "z_pion_neg",
         "z_kaon_pos", "z_kaon_neg", "pion_neg.*", "pion_pos.*", "pion_pos_E", "pion_neg_E",
    },
    # only run on mc
    mc_only=True,
)
def higgs_spins(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_higgs_decay_products](events, **kwargs)
    
    # seperate higgs decays to bb and tatau
    
    h0 = events.gen_higgs_decay.higgs[..., 0]
    h1 = events.gen_higgs_decay.higgs[:, 1]
    h_b = events.gen_higgs_decay.h_b[:, 0]
    h_tau = events.gen_higgs_decay.h_tau[:, 0]
  
    # calculate the cosine between two Higgs
    cos_h_h = (h0.px * h1.px + h0.py * h1.py + h0.pz * h1.pz) / (h0.p * h1.p) 


 

    b1 = events.gen_higgs_decay.b[:, 0]
    b2 = events.gen_higgs_decay.b[:, 1]

    cos_b_b = (b1.px * b2.px + b1.py * b2.py + b1.pz * b2.pz) / (b1.p * b2.p)

    tau1 = events.gen_higgs_decay.tau[:, 0]
    tau2 = events.gen_higgs_decay.tau[:, 1]

    cos_tau_tau = (tau1.px * tau2.px + tau1.py * tau2.py + tau1.pz * tau2.pz) / (tau1.p * tau2.p)
    
    
    
    # calculate the gamma of two Higgs
    #gamma_h_b = h_b.energy / h_b.mass 
    #gamma_h_tau = h_tau.energy / h_tau.mass

  

    # calculate the pt of two Higgs
    #pt_h_b = h_b.pt
    #pt_h_tau = h_tau.pt

   
    
    # calculate the cosine between Higgs and tau_pos/tau_neg
  
    tau_pos = events.gen_higgs_decay.tau_pos[:, 0]
    tau_neg = events.gen_higgs_decay.tau_neg[:, 0]


    cos_pos = (h_tau.px * tau_pos.px + h_tau.py * tau_pos.py + h_tau.pz * tau_pos.pz) / (h_tau.p * tau_pos.p)
    cos_neg = (h_tau.px * tau_neg.px + h_tau.py * tau_neg.py + h_tau.pz * tau_neg.pz) / (h_tau.p * tau_neg.p)
    # you have to fix it!!!!!
    # calculate energy fraction z for positive and negative tau
    #w_visible = events.gen_higgs_decay.w_visible_sum
    #w_pos_visible = w_visible[tau_pos]
    #w_neg_visible = w_visible[tau_neg]
   
    #z_po = ak.flatten(w_pos_visible.energy/tau_pos.energy, axis=None)
    #z_ne = ak.flatten(w_neg_visible.energy/tau_neg.energy, axis=None)
    #z_pos = ak.where(np.isnan(z_po),EMPTY_FLOAT,z_po)
    #z_neg = ak.where(np.isnan(z_ne),EMPTY_FLOAT,z_ne)
    

    # separate hadronic and leptonic z
    w_vis_had_pos = events.gen_higgs_decay.w_vis_had_pos[:, 0]
    w_vis_had_neg = events.gen_higgs_decay.w_vis_had_neg[:, 0]
    w_vis_lep_neg = events.gen_higgs_decay.w_vis_lep_neg[:, 0]
    w_vis_lep_pos = events.gen_higgs_decay.w_vis_lep_pos[:, 0]
    
    z_had_po = ak.flatten(w_vis_had_pos.energy/tau_pos.energy, axis=None)
    z_had_pos = ak.where(np.isnan(z_had_po),EMPTY_FLOAT,z_had_po)
    z_had_ne = ak.flatten(w_vis_had_neg.energy/tau_neg.energy, axis=None)
    z_had_neg = ak.where(np.isnan(z_had_ne),EMPTY_FLOAT,z_had_ne)
    
    z_lep_po = ak.flatten(w_vis_lep_pos.energy/tau_pos.energy, axis=None)
    z_lep_pos = ak.where(np.isnan(z_lep_po),EMPTY_FLOAT,z_lep_po)
    z_lep_ne = ak.flatten(w_vis_lep_neg.energy/tau_neg.energy, axis=None)
    z_lep_neg = ak.where(np.isnan(z_lep_ne),EMPTY_FLOAT,z_lep_ne)
    
    
    # pion_neg_neg = events.gen_higgs_decay.pion_neg_neg
    # pion_pos_pos = events.gen_higgs_decay.pion_pos_pos

    # pion_neg_pos = events.gen_higgs_decay.pion_neg_pos
    # pion_pos_neg = events.gen_higgs_decay.pion_pos_neg

    # z_pion_pos_po = ak.flatten(pion_pos_pos.energy/tau_pos.energy, axis=None)
    # z_pion_pos_pos = ak.where(np.isnan(z_pion_pos_po),EMPTY_FLOAT,z_pion_pos_po)

    # z_pion_neg_po = ak.flatten(pion_neg_pos.energy/tau_pos.energy, axis=None)
    # z_pion_neg_pos = ak.where(np.isnan(z_pion_neg_po),EMPTY_FLOAT,z_pion_neg_po)

    # z_pion_neg_ne = ak.flatten(pion_neg_neg.energy/tau_neg.energy, axis=None)
    # z_pion_neg_neg = ak.where(np.isnan(z_pion_neg_ne),EMPTY_FLOAT,z_pion_neg_ne)

    # z_pion_pos_ne = ak.flatten(pion_pos_neg.energy/tau_neg.energy, axis=None)
    # z_pion_pos_neg = ak.where(np.isnan(z_pion_pos_ne),EMPTY_FLOAT,z_pion_pos_ne)
    

    nu = events.gen_higgs_decay.nu
    nu_tau = nu[nu.pdgId == 16]
    anti_nu_tau = nu[nu.pdgId == -16]

    z_anti = ak.flatten(anti_nu_tau.energy/tau_pos.energy, axis=None)
    z_anti_nu = ak.where(np.isnan(z_anti),EMPTY_FLOAT,z_anti)
    
    z_tau = ak.flatten(nu_tau.energy/tau_neg.energy, axis=None)
    z_nu_tau = ak.where(np.isnan(z_tau),EMPTY_FLOAT,z_tau)
    
    
       
    

  

   

    events = set_ak_column_f32(events, "ts_cos_bb", cos_b_b)
    events = set_ak_column_f32(events, "ts_cos_hh", cos_h_h)
    events = set_ak_column_f32(events, "ts_cos_tautau", cos_tau_tau)
    #events = set_ak_column_f32(events, "ts_gamma_h_b", gamma_h_b )
    #events = set_ak_column_f32(events, "ts_gamma_h_tau", gamma_h_tau)
    #events = set_ak_column_f32(events, "ts_pt_bb", pt_h_b)
    #events = set_ak_column_f32(events, "ts_pt_tautau", pt_h_tau)
    events = set_ak_column_f32(events, "ts_cos_tau_neg", cos_neg)
    events = set_ak_column_f32(events, "ts_cos_tau_pos", cos_pos)
    # events = set_ak_column_f32(events, "ts_z_pos", z_pos)
    # events = set_ak_column_f32(events, "ts_z_neg", z_neg)
    events = set_ak_column_f32(events, "ts_z_had_neg", z_had_neg)
    events = set_ak_column_f32(events, "ts_z_had_pos", z_had_pos)
    events = set_ak_column_f32(events, "ts_z_lep_neg", z_lep_neg)
    events = set_ak_column_f32(events, "ts_z_lep_pos", z_lep_pos)
    events = set_ak_column_f32(events, "ts_z_anti_nu", z_anti_nu)
    events = set_ak_column_f32(events, "ts_z_nu_tau", z_nu_tau)



     

    return events

@producer(
    uses={
        gen_z_decay_products, 
        "GenPart.eta", "GenPart.pt", "GenPart.phi", "GenPart.mass",
    },
    produces={
        *optional_column("ts_tau_neg"),"z_pion_pos", "z_pion_neg", "z_kaon_pos", "z_kaon_neg", "pion_neg", "pion_pos",
    },
    # only run on mc
    mc_only=True,
)
def z_spins(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_z_decay_products](events, **kwargs) 


    

    tau_neg = events.gen_z_decay.tau_neg





    events = set_ak_column_f32(events, "ts_tau_neg", tau_neg)






    return events    
