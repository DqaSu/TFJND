from argparse import Namespace
from model.net import RDN


# @register('rdn')
def make_rdn(out_dim=1, G0=32, RDNkSize=3, RDNconfig='D', keep_prob=0.1, n_colors=3):
    args = Namespace()   # namespace is actually a dictionary
    args.out_dim = out_dim
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig
    args.keep_prob = keep_prob
    args.n_colors = n_colors
    return RDN(args)