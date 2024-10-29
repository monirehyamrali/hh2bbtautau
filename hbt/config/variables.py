# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT, flat_np_view


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="cos_tt",
        null_value=EMPTY_FLOAT,
        binning=(80, -1, 1),
        x_title=r"$cos_{\tau\tau}$",
    )

    config.add_variable(
        name="deltaR_tt",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 5),
        x_title=r"$\Delta R(\tau, \tau)$",
    )
    config.add_variable(
        name="deltaphi_tautau",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 5),
        x_title=r"$\Delta \phi_{(\tau, \tau)}$",
    )
    config.add_variable(
        name="deltaeta_tautau",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 5),
        x_title=r"$\Delta \eta_{(\tau, \tau)}$",
    )
    config.add_variable(
        name="jet1_btag",
        expression="Jet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 1),
        x_title=r"b-tag score of hardest jet",
    )

    config.add_variable(
        name="bjet1_btag",
        expression="BJet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 1),
        x_title=r"b-tag score of hardest b-tagged jet",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    config.add_variable(
        name="tau_pos_pt",
        expression="Tau_pos.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 400),
        unit="GeV",
        x_title="t$\tau^{+}$ $p_{T}$",
    )
    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    config.add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    config.add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    config.add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )
    config.add_variable(
        name="mjj",
        expression="mjj",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"m_{jj}",
    )
    config.add_variable(
        name="mtt",
        expression="mtt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$m_{\tau\tau}$",
    )
    config.add_variable(
        name="mbb",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )

    config.add_variable(
        name="ts_w0_mass",
        expression="ts_w0_mass",
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_{W0}$",
    )
    config.add_variable(
        name="ts_coslw",
        expression="ts_coslw",
        binning=(40, -1, 1),
        x_title=r"$cos_{lw}$",
    )
    config.add_variable(
        name="ts_cosnuw",
        expression="ts_cosnuw",
        binning=(40, -1, 1),
        x_title=r"$cos_{\nu w}$",
    )
    config.add_variable(
        name="ts_cos_bb",
        expression="ts_cos_bb",
        binning=(40, -1, 1),
        x_title=r"$cos_{b\bar{b}}$",
    )
    config.add_variable(
        name="ts_cos_hh",
        expression="ts_cos_hh",
        binning=(40, -1, 1),
        x_title=r"$cos_{hh}$",
    )
    config.add_variable(
        name="ts_cos_tautau",
        expression="ts_cos_tautau",
        binning=(40, -1, 1),
        x_title=r"$cos_{\tau\bar{\tau}}$",
    )
    config.add_variable(
        name="ts_cos_tau_neg",
        expression="ts_cos_tau_neg",
        binning=(40, -1, 1),
        x_title=r"$cos_{H\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_cos_tau_pos",
        expression="ts_cos_tau_pos",
        binning=(40, -1, 1),
        x_title=r"$cos_{H\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_had_neg",
        expression="ts_z_had_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{vis-had}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_had_pos",
        expression="ts_z_had_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{vis-had}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_lep_neg",
        expression="ts_z_lep_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{vis-lep}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_lep_pos",
        expression="ts_z_lep_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{vis-lep}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_pion_pos_neg",
        expression="ts_z_pion_pos_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_pion_neg_pos",
        expression="ts_z_pion_neg_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_pion_pos_pos",
        expression="ts_z_pion_pos_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_pion_neg_neg",
        expression="ts_z_pion_neg_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_anti_nu",
        expression="ts_z_anti_nu",
        binning=(40, 0, 2.),
        x_title=r"$E_{\nu}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_nu_tau",
        expression="ts_z_nu_tau",
        binning=(40, 0, 2.),
        x_title=r"$E_{\nu}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_pion_zero_neg",
        expression="ts_z_pion_zero_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi0}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="ts_z_pion_zero_pos",
        expression="ts_z_pion_zero_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi0}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_pi_pos",
        expression="ts_z_pi_pos",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi^0,\pi^{pos}}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_pi_neg",
        expression="ts_z_pi_neg",
        binning=(40, 0, 2.),
        x_title=r"$E_{\pi^0,\pi^{neg}}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="z_pion_neg",
        expression="z_pion_neg",
        binning=(40, 0, 1.),
        x_title=r"$E_{\pi^{neg}}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="z_pion_pos",
        expression="z_pion_pos",
        binning=(40, 0, 1.),
        x_title=r"$E_{\pi^{pos}}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="z_kaon_pos",
        expression="z_kaon_pos",
        binning=(40, 0, 1.),
        x_title=r"$E_{K^{pos}}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="z_kaon_neg",
        expression="z_kaon_neg",
        binning=(40, 0, 1.),
        x_title=r"$E_{K^{neg}}/E_{\tau^{neg}}$",
    )
    #remove later
    config.add_variable(
        name="ts_tau_neg",
        expression="ts_tau_neg",
        binning=(40, 0, 2.),
        x_title=r"$\tau$",
    )
    config.add_variable(
        name="ts_t",
        expression="ts_t",
        binning=(40, 0, 2.),
        x_title=r"$t$",
    )
    config.add_variable(
        name="z_pion_det_neg",
        expression="z_pion_det_neg",
        binning=(40, 0, 1.),
        x_title=r"$E_{\pi^{neg}}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="pion_det_neg.zfrac",
        expression="pion_det_neg.zfrac",
        binning=(40, 0, 1.),
        x_title=r"$E_{\pi^{neg}}/E_{\tau^{neg}}$ (direct)",
    )
    config.add_variable(
        name="z_pion_det_pos",
        expression="z_pion_det_pos",
        binning=(40, 0, 1.),
        x_title=r"$E_{\pi^{pos}}/E_{\tau^{pos}}$",
    )
    config.add_variable(
       name="pion_pos_E",
       expression="pion_pos_E",
       binning=(40, 0.0, 400.0),
       unit="GeV",
       x_title=r"$E_{\pi^{pos}}$",
    )
    config.add_variable(
       name="pion_neg_E",
       expression="pion_neg_E",
       binning=(40, 0.0, 400.0),
       unit="GeV",
       x_title=r"$E_{\pi^{neg}}$",
    )

    config.add_variable(
       name="pion_pos.E",
       expression="pion_pos.E",
       binning=(40, 0.0, 400.0),
       unit="GeV",
       x_title=r"$E_{\pi^{pos}} (direct)$",
    )
    config.add_variable(
       name="pion_neg.E",
       expression="pion_neg.E",
       binning=(40, 0.0, 400.0),
       null_value=EMPTY_FLOAT,
       unit="GeV",
       x_title=r"$E_{\pi^{neg}} (direct)$",
    )

    config.add_variable(
       name="pion_det_pos.phi",
       expression="pion_det_pos.phi",
       binning=(40, -4, 4),
       null_value=EMPTY_FLOAT,
       unit="rad",
       x_title=r"$\phi^{det}_{\pi^{pos}} (direct)$",
    )
    config.add_variable(
       name="pion_det_neg.phi",
       expression="pion_det_neg.phi",
       binning=(40, -4, 4),
       null_value=EMPTY_FLOAT,
       unit="rad",
       x_title=r"$\phi^{det}_{\pi^{neg}} (direct)$",
    )


    config.add_variable(
       name="pion_det_pos_E",
       expression="pion_det_pos_E",
       binning=(40, 0.0, 400.0),
       unit="GeV",
       x_title=r"$E^{det}_{\pi^{pos}}$",
    )
    config.add_variable(
       name="pion_det_neg_E",
       expression="pion_det_neg_E",
       binning=(40, 0.0, 400.0),
       unit="GeV",
       x_title=r"$E^{det}_{\pi^{neg}}$",
    )
    # config.add_variable(
    #     name="pion_neg.zfrac",
    #     expression="pion_neg.zfrac",
    #     binning=(40, 0, 1.0),
    #     x_title=r"$E_{\pi^{neg}}/E_{\tau^{neg}}$ ",
    # )
    # config.add_variable(
    #     name="pion_pos.zfrac",
    #     expression="pion_pos.zfrac",
    #     binning=(40, 0, 1.0),
    #     x_title=r"$E_{\pi^{pos}}/E_{\tau^{pos}}$ ",
    # )
    config.add_variable(
        name="pion_neg.zfrac",
        expression="pion_neg.zfrac",
        binning=(2, 0, 1.0),
        x_title=r"$E_{\pi^{neg}}/E_{\tau^{neg}}$ ",
    )
    config.add_variable(
        name="pion_pos.zfrac",
        expression="pion_pos.zfrac",
        binning=(2, 0, 1.0),
        x_title=r"$E_{\pi^{pos}}/E_{\tau^{pos}}$ ",
    )
    config.add_variable(
        name="ts_z_pos",
        expression="ts_z_pos",
        binning=(40, 0, 1.),
        x_title=r"$E^{pos}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="ts_z_neg",
        expression="ts_z_neg",
        binning=(40, 0, 1.),
        x_title=r"$E^{neg}/E_{\tau^{neg}}$",
    )
    config.add_variable(
        name="z_pos",
        expression="z_pos",
        binning=(40, 0, 1.),
        x_title=r"$E^{pos}/E_{\tau^{pos}}$",
    )
    config.add_variable(
        name="z_neg",
        expression="z_neg",
        binning=(40, 0, 1.),
        x_title=r"$E^{neg}/E_{\tau^{neg}}$",
    )

    def z_rec_dm_expression(charge, decay_mode):
        def expr(events):
            z = flat_np_view(events[f"z_rec_{charge}"], axis=0)
            dm = events[f"dm_{charge}"]
            z[dm != decay_mode] = EMPTY_FLOAT
            return z
        return expr

    for dm in [-1, 0, 1, 10, 11]:
        dm_str = str(dm).replace("-", "m")
        config.add_variable(
            name=f"z_rec_pos_dm{dm_str}",
            expression=z_rec_dm_expression("pos", dm),
            binning=(40, 0, 1.005),
            x_title=rf"$E^{{pos}}/E_{{\tau^{{pos}}}} (DM{dm},rec)$",
            aux={"inputs": ["z_rec_pos", "dm_pos"]},
        )
        config.add_variable(
            name=f"z_rec_neg_dm{dm_str}",
            expression=z_rec_dm_expression("neg", dm),
            binning=(40, 0, 1.005),
            x_title=rf"$E^{{neg}}/E_{{\tau^{{neg}}}} (DM{dm},rec)$",
            aux={"inputs": ["z_rec_neg", "dm_neg"]},
        )


    def z_gen_dm_expression(charge, decay_mode):
        def expr(events):
            z = flat_np_view(events[f"z_gen_{charge}"], axis=0)
            dm = events[f"dm_{charge}"]
            z[dm != decay_mode] = EMPTY_FLOAT
            return z
        return expr

    for dm in [-1, 0, 1, 10, 11]:
        dm_str = str(dm).replace("-", "m")
        config.add_variable(
            name=f"z_gen_pos_dm{dm_str}",
            expression=z_gen_dm_expression("pos", dm),
            binning=(40, 0, 1.005),
            x_title=rf"$E^{{pos}}/E_{{\tau^{{pos}}}} (DM{dm},gen)$",
            aux={"inputs": ["z_gen_pos", "dm_pos"]},

        )
        config.add_variable(
            name=f"z_gen_neg_dm{dm_str}",
            expression=z_gen_dm_expression("neg", dm),
            binning=(40, 0, 1.005),
            x_title=rf"$E^{{neg}}/E_{{\tau^{{neg}}}} (DM{dm},gen)$",
            aux={"inputs": ["z_gen_neg", "dm_neg"]},
        )
        




