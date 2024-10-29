# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
import functools

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from hbt.production.tautauNN import tautauNN
from hbt.production.gen_top_decay import gen_top_decay_products
from hbt.production.gen_higgs_decay import gen_higgs_decay_products
from hbt.production.gen_z_decay import gen_z_decay_products
from hbt.production.tau_zfrac import top_nu, higgs_nu, z_nu

ak = maybe_import("awkward")
np = maybe_import("numpy")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@producer(
    uses={
        category_ids, normalization_weights, tautauNN,  attach_coffea_behavior,
        "Tau.pt", "Tau.eta", "Tau.mass", "Tau.charge", "Tau.decayMode", "Tau.phi", "tau_nus.*",
    },
    produces={
        category_ids, normalization_weights, tautauNN, 
        "z_rec_neg", "z_rec_pos", "z_gen_neg", "z_gen_pos", "dm_neg", "dm_pos",
        
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
)
def z_fractions(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)
    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)
        events = self[tautauNN](events, **kwargs)
         
    print("debugger at the right postition")
    z_rec_pos = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
    z_rec_neg = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
    z_gen_pos = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
    z_gen_neg = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
    dm_pos = EMPTY_FLOAT * np.ones(len(events), dtype=np.int32)
    dm_neg = EMPTY_FLOAT * np.ones(len(events), dtype=np.int32)

    # from IPython import embed; embed()

    #get regressed neutrino components
    px_1 = events.tautauNN_regression_output[:,0]
    py_1 = events.tautauNN_regression_output[:,1]
    pz_1 = events.tautauNN_regression_output[:,2]
    p_1 = np.sqrt(px_1**2 + py_1**2 + pz_1**2)
    px_2 = events.tautauNN_regression_output[:,3]
    py_2 = events.tautauNN_regression_output[:,4]    
    pz_2 = events.tautauNN_regression_output[:,5]
    p_2 = np.sqrt(px_2**2 + py_2**2 + pz_2**2)

    # gen neutrinos
    tau_nus = events.tau_nus
    p_neg = np.sqrt((tau_nus.x[:,:1])**2 + (tau_nus.y[:,:1])**2 + (tau_nus.z[:,:1])**2)
    p_pos = np.sqrt((tau_nus.x[:, 1:])**2 + (tau_nus.y[:, 1:])**2 + (tau_nus.z[:, 1:])**2)

    get_z = lambda e, p_nu: e / (e + p_nu)

    # e neg, tau pos
    ele_neg = np.asarray(ak.num(events.Electron, axis=1) == 1)
    ele_neg[ele_neg] = events[ele_neg].Electron[:, 0].charge == -1
    z_rec_neg[ele_neg] = get_z(events.Electron.energy[ele_neg, 0], p_1[ele_neg])
    dm_neg[ele_neg] = -1
    z_rec_pos[ele_neg] = get_z(events.Tau.energy[ele_neg, 0], p_2[ele_neg])
    dm_pos[ele_neg] = events.Tau.decayMode[ele_neg, 0]

    # e pos, tau neg
    ele_pos = np.asarray(ak.num(events.Electron, axis=1) == 1)
    ele_pos[ele_pos] = events[ele_pos].Electron[:, 0].charge == 1
    z_rec_pos[ele_pos] = get_z(events.Electron.energy[ele_pos, 0], p_1[ele_pos])
    dm_pos[ele_pos] = -1
    z_rec_neg[ele_pos] = get_z(events.Tau.energy[ele_pos, 0], p_2[ele_pos])
    dm_neg[ele_pos] = events.Tau.decayMode[ele_pos, 0]

    # mu neg, tau pos
    mu_neg = np.asarray(ak.num(events.Muon, axis=1) == 1)
    mu_neg[mu_neg] = events[mu_neg].Muon[:, 0].charge == -1
    z_rec_neg[mu_neg] = get_z(events.Muon.energy[mu_neg, 0], p_1[mu_neg])
    dm_neg[mu_neg] = -1
    z_rec_pos[mu_neg] = get_z(events.Tau.energy[mu_neg, 0], p_2[mu_neg])
    dm_pos[mu_neg] = events.Tau.decayMode[mu_neg, 0]

    # mu pos, tau neg
    mu_pos = np.asarray(ak.num(events.Muon, axis=1) == 1)
    mu_pos[mu_pos] = events[mu_pos].Muon[:, 0].charge == 1
    z_rec_pos[mu_pos] = get_z(events.Muon.energy[mu_pos, 0], p_1[mu_pos])
    dm_pos[mu_pos] = -1
    z_rec_neg[mu_pos] = get_z(events.Tau.energy[mu_pos, 0], p_2[mu_pos])
    dm_neg[mu_pos] = events.Tau.decayMode[mu_pos, 0]
    
    # tau neg, tau pos
    tau1_neg = np.asarray((ak.num(events.Electron, axis=1) + ak.num(events.Muon, axis=1)) != 1)
    tau1_neg[tau1_neg] = events[tau1_neg].Tau[:, 0].charge == -1
    z_rec_neg[tau1_neg] = get_z(events.Tau.energy[tau1_neg, 0], p_1[tau1_neg])
    dm_neg[tau1_neg] = events.Tau.decayMode[tau1_neg, 0]
    z_rec_pos[tau1_neg] = get_z(events.Tau.energy[tau1_neg, 1], p_2[tau1_neg])
    dm_pos[tau1_neg] = events.Tau.decayMode[tau1_neg, 1]

    # tau pos, tau neg
    tau1_pos = np.asarray((ak.num(events.Electron, axis=1) + ak.num(events.Muon, axis=1)) != 1)
    tau1_pos[tau1_pos] = events[tau1_pos].Tau[:, 0].charge == 1
    z_rec_pos[tau1_pos] = get_z(events.Tau.energy[tau1_pos, 0], p_1[tau1_pos])
    dm_pos[tau1_pos] = events.Tau.decayMode[tau1_pos, 0]
    z_rec_neg[tau1_pos] = get_z(events.Tau.energy[tau1_pos, 1], p_2[tau1_pos])
    dm_neg[tau1_pos] = events.Tau.decayMode[tau1_pos, 1]

    # gen level z

    ele_neg = np.asarray(ak.num(events.Electron, axis=1) == 1)
    ele_neg[ele_neg] = events[ele_neg].Electron[:, 0].charge == -1
    z_gen_neg[ele_neg] = get_z(events.Electron.energy[ele_neg, 0], ak.flatten(p_neg[ele_neg]))
    dm_neg[ele_neg] = -1
    z_gen_pos[ele_neg] = get_z(events.Tau.energy[ele_neg, 0], ak.flatten(p_neg[ele_neg]))
    dm_pos[ele_neg] = events.Tau.decayMode[ele_neg, 0]

    # e pos, tau neg
    ele_pos = np.asarray(ak.num(events.Electron, axis=1) == 1)
    ele_pos[ele_pos] = events[ele_pos].Electron[:, 0].charge == 1
    z_gen_pos[ele_pos] = get_z(events.Electron.energy[ele_pos, 0], ak.flatten(p_pos[ele_pos]))
    dm_pos[ele_pos] = -1
    z_gen_neg[ele_pos] = get_z(events.Tau.energy[ele_pos, 0], ak.flatten(p_pos[ele_pos]))
    dm_neg[ele_pos] = events.Tau.decayMode[ele_pos, 0]

    # mu neg, tau pos
    mu_neg = np.asarray(ak.num(events.Muon, axis=1) == 1)
    mu_neg[mu_neg] = events[mu_neg].Muon[:, 0].charge == -1
    z_gen_neg[mu_neg] = get_z(events.Muon.energy[mu_neg, 0], ak.flatten(p_neg[mu_neg]))
    dm_neg[mu_neg] = -1
    z_gen_pos[mu_neg] = get_z(events.Tau.energy[mu_neg, 0], ak.flatten(p_neg[mu_neg]))
    dm_pos[mu_neg] = events.Tau.decayMode[mu_neg, 0]

    # mu pos, tau neg
    mu_pos = np.asarray(ak.num(events.Muon, axis=1) == 1)
    mu_pos[mu_pos] = events[mu_pos].Muon[:, 0].charge == 1
    z_gen_pos[mu_pos] = get_z(events.Muon.energy[mu_pos, 0], ak.flatten(p_pos[mu_pos]))
    dm_pos[mu_pos] = -1
    z_gen_neg[mu_pos] = get_z(events.Tau.energy[mu_pos, 0], ak.flatten(p_pos[mu_pos]))
    dm_neg[mu_pos] = events.Tau.decayMode[mu_pos, 0]
    
    # tau neg, tau pos
    tau1_neg = np.asarray((ak.num(events.Electron, axis=1) + ak.num(events.Muon, axis=1)) != 1)
    tau1_neg[tau1_neg] = events[tau1_neg].Tau[:, 0].charge == -1
    z_gen_neg[tau1_neg] = get_z(events.Tau.energy[tau1_neg, 0], ak.flatten(p_neg[tau1_neg]))
    dm_neg[tau1_neg] = events.Tau.decayMode[tau1_neg, 0]
    z_gen_pos[tau1_neg] = get_z(events.Tau.energy[tau1_neg, 1], ak.flatten(p_neg[tau1_neg]))
    dm_pos[tau1_neg] = events.Tau.decayMode[tau1_neg, 1]

    # tau pos, tau neg
    tau1_pos = np.asarray((ak.num(events.Electron, axis=1) + ak.num(events.Muon, axis=1)) != 1)
    tau1_pos[tau1_pos] = events[tau1_pos].Tau[:, 0].charge == 1
    z_gen_pos[tau1_pos] = get_z(events.Tau.energy[tau1_pos, 0], ak.flatten(p_pos[tau1_pos]))
    dm_pos[tau1_pos] = events.Tau.decayMode[tau1_pos, 0]
    z_gen_neg[tau1_pos] = get_z(events.Tau.energy[tau1_pos, 1], ak.flatten(p_pos[tau1_pos]))
    dm_neg[tau1_pos] = events.Tau.decayMode[tau1_pos, 1]
    #from IPython import embed; embed(header="z_frac")

    events = set_ak_column(events, "z_rec_neg", z_rec_neg)
    events = set_ak_column(events, "z_rec_pos", z_rec_pos)
    events = set_ak_column(events, "z_gen_neg", z_gen_neg)
    events = set_ak_column(events, "z_gen_pos", z_gen_pos)
    events = set_ak_column(events, "dm_neg", dm_neg)
    events = set_ak_column(events, "dm_pos", dm_pos)
    #from IPython import embed; embed()
    # Dec_Mode = events.Tau.decayMode
    # m0 = Dec_Mode == 0
    # m1 = Dec_Mode == 1
    # neg = events.Tau.charge[m1] == -1
    # pos = events.Tau.charge[m1] == 1

    # DM_neg = Dec_Mode[neg]
    # DM_pos = Dec_Mode[pos]
    # #from IPython import embed; embed()
    # E = np.sqrt((events.Tau.pt[m1])**2 * (1+(np.sinh(events.Tau.eta[m1]))**2) + (events.Tau.mass[m1])** 2)

    # E_pos = ak.fill_none(E[pos], [], axis=0)
    # E_neg = ak.fill_none(E[neg], [], axis=0)
    

    # # px_1 = events.tautauNN_regression_output[:,0]
    # # py_1 = events.tautauNN_regression_output[:,1]
    # # pz_1 = events.tautauNN_regression_output[:,2]
    
    # # px_2 = events.tautauNN_regression_output[:,3]
    # # py_2 = events.tautauNN_regression_output[:,4]    
    # # pz_2 = events.tautauNN_regression_output[:,5]

    # # p_neg = np.sqrt( px_1**2 + py_1**2 + pz_1**2)
    # # p_pos = np.sqrt( px_2**2 + py_2**2 + pz_2**2)

    # tau_nus = events.tau_nus
    # p_neg = np.sqrt((tau_nus.x[:,:1])**2 + (tau_nus.y[:,:1])**2 + (tau_nus.z[:,:1])**2)
    # p_pos = np.sqrt((tau_nus.x[:, 1:])**2 + (tau_nus.y[:, 1:])**2 + (tau_nus.z[:, 1:])**2)

    
    # z_ne = E_neg / (E_neg +  p_neg)
    # z_neg = ak.where(np.isnan(z_ne),EMPTY_FLOAT,z_ne)
  

    # z_po = E_pos / (E_pos +  p_pos)
    # z_pos = ak.where(np.isnan(z_po),EMPTY_FLOAT,z_po)
   

    # #print("debugger at the right postition")
    # #from IPython import embed; embed()
    # events = set_ak_column_f32(events, "z_neg", z_neg)
    # events = set_ak_column_f32(events, "z_pos", z_pos)
    # events = set_ak_column_f32(events, "dm_neg", DM_neg)
    # events = set_ak_column_f32(events, "dm_pos", DM_pos)

    # version for gen level: dev2_tau_nu
    # version for Higgs & tt rec. : dev3_tau_nu_5
    # version for drell-yan rec. : dev1_tau_nu

    # # get e
    # e = ak.num(events.Electron) == 1
    # e_neg = events.Electron[e].charge == -1
    # z_g_neg = get_z(events.Tau.energy[events.Electron.charge == -1][e], p_neg[events.Electron.charge == -1][e])
    # e_pos = events.Electron[e].charge == 1
    # z_g_pos = get_z(events.Tau.energy[events.Electron.charge == 1][e], p_pos[events.Electron.charge == 1][e])

    # # get mu
    # mu = ak.num(events.Muon) == 1
    # mu_neg = events.Muon[mu].charge == -1
    # z_g_neg = get_z(events.Tau.energy[events.Muon.charge == -1][mu], p_neg[events.Muon.charge == -1][mu])
    # mu_pos = events.Muon[mu].charge == 1
    # z_g_pos = get_z(events.Tau.energy[events.Muon.charge == 1][mu], p_pos[events.Muon.charge == 1][mu])






















    return events
