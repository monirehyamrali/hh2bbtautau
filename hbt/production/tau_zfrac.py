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
        "Tau.pt", "Tau.eta", "Tau.mass", "Tau.charge", "Tau.decayMode", "Tau.phi",
    },
    produces={
        gen_top_decay_products,
        *optional_column("ts_z_neg", "ts_z_pos"),
    
    },
    # only run on mc
    mc_only=True,
)
def top_nu(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_top_decay_products](events, **kwargs)
    
    
    Dec_Mode = events.Tau.decayMode
    # 3 charged currents
    m10 = Dec_Mode == 10

    # 1 charged current & 1 neutral
    m1 = Dec_Mode == 1

    # only 1 charged current
    m0 = Dec_Mode == 0
    
    # mask for events with only 2 Taus
    mask_2t = ak.num(events.Tau.pt) ==  2
    Tau_2 = ak.mask(events.Tau.charge, mask_2t)

    # mask for 2 different charges
    mask_ch = ak.sum(events.Tau.charge, axis=-1) == 0
    

    # tau_neg = events.gen_top_decay[:, 12]
    # tau_pos = events.gen_top_decay[:, 13]

    

    tau_nus = events.tau_nus
    p_nu_neg = np.sqrt((tau_nus.x[:,:1])**2 + (tau_nus.y[:,:1])**2 + (tau_nus.z[:,:1])**2)
    p_nu_pos = np.sqrt((tau_nus.x[:, 1:])**2 + (tau_nus.y[:, 1:])**2 + (tau_nus.z[:, 1:])**2)

    #from IPython import embed; embed()

    # reconstruct energy
    E_ne = np.sqrt((events.Tau.pt[events.Tau.charge[m0] == -1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge[m0] == -1]))**2) + (events.Tau.mass[events.Tau.charge[m0] == -1])**2)
    E_po = np.sqrt((events.Tau.pt[events.Tau.charge[m0] == 1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge[m0] == 1]))**2) + (events.Tau.mass[events.Tau.charge[m0] == 1])**2)
    
    E_negg = ak.mask(E_ne, mask_2t)
    En = ak.mask(E_negg, mask_ch)
    E_neg = ak.fill_none(En, [], axis=0)
    

    E_poss = ak.mask(E_po, mask_2t)
    Ep = ak.mask(E_poss, mask_ch)
    E_pos = ak.fill_none(Ep, [], axis=0)
    
    
    z_ne = E_neg / (E_neg +  p_nu_neg)
    z_neg = ak.where(np.isnan(z_ne),EMPTY_FLOAT,z_ne)

    z_po = E_pos / (E_pos +  p_nu_pos)
    z_pos = ak.where(np.isnan(z_po),EMPTY_FLOAT,z_po)



    events = set_ak_column_f32(events, "ts_z_neg", z_neg)
    events = set_ak_column_f32(events, "ts_z_pos", z_pos)

    

    return events






@producer(
    uses={
        gen_higgs_decay_products,
        "Tau.pt", "Tau.eta", "Tau.mass", "Tau.charge", "Tau.decayMode", "Tau.phi",
    },
    produces={
        # gen_higgs_decay_products,
        *optional_column("ts_z_neg", "ts_z_pos"),
    },
    # only run on mc
    mc_only=True,
)
def higgs_nu(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_higgs_decay_products](events, **kwargs)
    

    
    Dec_Mode = events.Tau.decayMode
    
    # 3 charged currents
    m10 = Dec_Mode == 10

    # 1 charged current & 1 neutral
    m1 = Dec_Mode == 1

    # only 1 charged current
    m0 = Dec_Mode == 0

    # mask for events with only 2 Taus
    mask_2t = ak.num(events.Tau.pt) ==  2
    Tau_2 = ak.mask(events.Tau.pt, mask_2t)

    # mask for 2 different charges
    mask_ch = ak.sum(events.Tau.charge, axis=-1) == 0

    # tau_pos = events.gen_higgs_decay.tau_pos[:, 0]
    # tau_neg = events.gen_higgs_decay.tau_neg[:, 0]
    #from IPython import embed; embed()
    # eventuell mit Tobias Daten ersetzen
    tau_nus = events.tau_nus
    p_nu_neg = np.sqrt((tau_nus.x[:,0])**2 + (tau_nus.y[:,0])**2 + (tau_nus.z[:,0])**2)
    p_nu_pos = np.sqrt((tau_nus.x[:,1])**2 + (tau_nus.y[:,1])**2 + (tau_nus.z[:,1])**2)
    from IPython import embed; embed()
    # reconstruct energy
    E_ne = np.sqrt((events.Tau.pt[events.Tau.charge == -1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge == -1]))**2) + (events.Tau.mass[events.Tau.charge[m1] == -1])**2)
    E_po = np.sqrt((events.Tau.pt[events.Tau.charge == 1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge == 1]))**2) + (events.Tau.mass[events.Tau.charge[m1] == 1])**2)
    
    E_negg = ak.mask(E_ne, mask_2t)
    En = ak.mask(E_negg, mask_ch)
    E_neg = ak.fill_none(En, [], axis=0)
    #from IPython import embed; embed()
    E_poss = ak.mask(E_po, mask_2t)
    Ep = ak.mask(E_poss, mask_ch)
    E_pos = ak.fill_none(Ep, [], axis=0)


    z_ne = E_neg / (E_neg +  p_nu_neg)
    z_neg = ak.where(np.isnan(z_ne),EMPTY_FLOAT,z_ne)

    z_po = E_pos / (E_pos +  p_nu_pos)
    z_pos = ak.where(np.isnan(z_po),EMPTY_FLOAT,z_po)
    
       
    events = set_ak_column_f32(events, "ts_z_neg", z_neg)
    events = set_ak_column_f32(events, "ts_z_pos", z_pos)


    return events






@producer(
    uses={
        gen_z_decay_products, 
        "Tau.pt", "Tau.eta", "Tau.mass", "Tau.charge", "Tau.decayMode", "Tau.phi",
    },
    produces={
        *optional_column("ts_z_neg", "ts_z_pos"), gen_z_decay_products,
    },
    # only run on mc
    mc_only=True,
)
def z_nu(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = self[gen_z_decay_products](events, **kwargs) 


    
    Dec_Mode = events.Tau.decayMode
    
    # 3 charged currents
    m10 = Dec_Mode == 10
    m11 = Dec_Mode == 11

    # 1 charged current & 1 neutral
    m1 = Dec_Mode == 1

    # only 1 charged current
    m0 = Dec_Mode == 0

    # mask for events with only 2 Taus
    mask_2t = ak.num(events.Tau.pt) ==  2
    Tau_2 = ak.mask(events.Tau.charge, mask_2t)

    # mask for 2 different charges
    mask_ch = ak.sum(events.Tau.charge, axis=-1) == 0


    tau_neg = events.gen_z_decay.tau_neg
    tau_pos = events.gen_z_decay.tau_pos
    tau_nus = events.tau_nus


    p_nu_neg = np.sqrt((tau_nus.x[:,:1])**2 + (tau_nus.y[:,:1])**2 + (tau_nus.z[:,:1])**2)
    p_nu_pos = np.sqrt((tau_nus.x[:, 1:])**2 + (tau_nus.y[:, 1:])**2 + (tau_nus.z[:, 1:])**2)
    

    # reconstruct energy
    E_ne = np.sqrt((events.Tau.pt[events.Tau.charge[m0] == -1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge[m0] == -1]))**2) + (events.Tau.mass[events.Tau.charge[m0] == -1])**2)
    E_po = np.sqrt((events.Tau.pt[events.Tau.charge[m0] == 1])**2 * (1+(np.sinh(events.Tau.eta[events.Tau.charge[m0] == 1]))**2) + (events.Tau.mass[events.Tau.charge[m0] == 1])**2)


    E_negg = ak.mask(E_ne, mask_2t)
    En = ak.mask(E_negg, mask_ch)
    E_neg = ak.fill_none(En, [], axis=0)
    
    E_poss = ak.mask(E_po, mask_2t)
    Ep = ak.mask(E_poss, mask_ch)
    E_pos = ak.fill_none(Ep, [], axis=0)


    z_ne = E_neg / (E_neg +  p_nu_neg)
    z_neg = ak.where(np.isnan(z_ne),EMPTY_FLOAT,z_ne)

    z_po = E_pos / (E_pos +  p_nu_pos)
    z_pos = ak.where(np.isnan(z_po),EMPTY_FLOAT,z_po)
    
       
    events = set_ak_column_f32(events, "ts_z_neg", z_neg)
    events = set_ak_column_f32(events, "ts_z_pos", z_pos)




    return events    
