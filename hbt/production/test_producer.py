"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from columnflow.production.util import attach_coffea_behavior


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)  

@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mjj",
    },
) 
def invariantmass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    # from IPython import embed; embed()
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))  
    mjj = (events.Jet[:,0] + events.Jet[:,1]).mass 
    events = set_ak_column(events, "mjj", ak.fill_none(mjj, EMPTY_FLOAT))

    return events   

@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mtt",
    },
) 
def invariantmasstau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    #from IPython import embed; embed()
    mtt = events.Tau[:,:2].sum(axis=1)
    mtt = mtt.mass 
    #from IPython import embed; embed()
    mask = ak.count(events.Tau.pt, axis=1) > 1 
    mtt = ak.mask(mtt, mask)
    events = set_ak_column_f32(events, "mtt", ak.fill_none(mtt, EMPTY_FLOAT))

    # alternative to above code:
    # ...
    # mtt = mtt.mass
    # mask = ak.count(events.Tau.pt, axis=1) > 1
    # events = set_ak_column_f32(events, "mtt", ak.where(mask, mtt, EMPTY_FLOAT))
    return events 


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass", 
        attach_coffea_behavior,
    },
    produces={
        "mbb",
    },
) 
def invariantmassbb(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    #from IPython import embed; embed()
    mbb = events.BJet[:, :2].sum(axis=1)
    mbb = mbb.mass 
    #from IPython import embed; embed()
    mask = ak.count(events.BJet.pt, axis=1) > 1 
    mbb = ak.mask(mbb, mask)
    events = set_ak_column_f32(events, "mbb", ak.fill_none(mbb, EMPTY_FLOAT))
    return events  


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "cos_tt"
    },
)
def cosinustautau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)

    tau_particles = events.Tau
 
    # Select tau particles with at least 2 taus.
    mask = ak.count(tau_particles.pt, axis=1) > 1
    tau_particles = ak.mask(tau_particles, mask)
    # from IPython import embed; embed()

    # Calculate the cosine of the angle between all possible combinations of tau pairs in the transverse plain.
    dot_prod = tau_particles[:,0].x*tau_particles[:,1].x + tau_particles[:,0].y*tau_particles[:,1].y
    norm = tau_particles[:, 0].pt * tau_particles[:, 1].pt
    cos_tt = dot_prod/(norm)

    events = set_ak_column_f32(events, "cos_tt", ak.fill_none(cos_tt, EMPTY_FLOAT))
    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "deltaR_tt"
    },
)
def deltaRtautau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)

    tau_particles = events.Tau
 
    # Select tau particles with at least 2 taus.
    mask = ak.count(tau_particles.pt, axis=1) > 1
    tau_particles = ak.mask(tau_particles, mask)
    # from IPython import embed; embed()

    deltar = tau_particles[:, 0].delta_r(tau_particles[:, 1])

    events = set_ak_column_f32(events, "deltaR_tt", ak.fill_none(deltar, EMPTY_FLOAT))
    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "deltaphi_tautau"
    },
)

def deltaphitautau(self:producer, events: ak.Array, **kwards) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwards)
    tau_particles = events.Tau
    #from IPython import embed; embed()

    mask = ak.count(tau_particles.phi, axis=1) > 1
    tau_particles = ak.mask(tau_particles, mask)
    tau1phi = tau_particles.phi[:, 0]
    tau2phi = tau_particles.phi[:, 1]
    deltaphi = np.abs(tau1phi - tau2phi)

    events = set_ak_column_f32(events, "deltaphi_tautau", ak.fill_none(deltaphi, EMPTY_FLOAT))
    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "deltaeta_tautau"
    },
)

def deltaetatautau(self:producer, events: ak.Array, **kwards) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwards)
    tau_particles = events.Tau
    #from IPython import embed; embed()

    mask = ak.count(tau_particles.eta, axis=1) > 1
    tau_particles = ak.mask(tau_particles, mask)
    tau1eta = tau_particles.eta[:, 0]
    tau2eta = tau_particles.eta[:, 1]
    deltaeta = np.abs(tau1eta - tau2eta)

    events = set_ak_column_f32(events, "deltaeta_tautau", ak.fill_none(deltaeta, EMPTY_FLOAT))
    return events


