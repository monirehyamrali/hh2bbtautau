# coding: utf-8

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

from hbt.production.test_producer import (cosinustautau, deltaRtautau,
        invariantmassbb, invariantmasstau, invariantmass, deltaphitautau, deltaetatautau
    )

ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariantmass, invariantmasstau, 
        invariantmassbb, cosinustautau, deltaRtautau, deltaphitautau, deltaetatautau,
    },
    produces={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariantmass, invariantmasstau,
         invariantmassbb, cosinustautau, deltaRtautau, deltaphitautau, deltaetatautau,

    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = self[category_ids](events, **kwargs)

   

    # features
    events = self[features](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # normalized pdf weight
        events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights](events, **kwargs)

        # tau weights
        events = self[tau_weights](events, **kwargs)

        # electron weights
        events = self[electron_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

        # trigger weights
        events = self[trigger_weights](events, **kwargs)

        #custom producer
        events = self[invariantmass](events, **kwargs)

        #custom producer
        events = self[invariantmasstau](events, **kwargs)

         #custom producer
        events = self[invariantmassbb](events, **kwargs)

         #custom producer
        events = self[cosinustautau](events, **kwargs)
         
         #custom producer
        events = self[deltaRtautau](events, **kwargs)

         #custom producer
        events = self[deltaphitautau](events, **kwargs)

         #custom producer
        events = self[deltaetatautau](events, **kwargs)



    return events




@producer(
    uses={
        category_ids,
    },
    produces={
        category_ids,
    },
)
def empty(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    # from IPython import embed
    # embed()
    events = self[category_ids](events, **kwargs)

    # # mc-only weights
    # if self.dataset_inst.is_mc:
    #     # normalization weights
    #     events = self[normalization_weights](events, **kwargs)

    #     # normalized pdf weight
    #     events = self[normalized_pdf_weight](events, **kwargs)

    #     # normalized renorm./fact. weight
    #     events = self[normalized_murmuf_weight](events, **kwargs)

    #     # normalized pu weights
    #     events = self[normalized_pu_weight](events, **kwargs)

    return events