import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("input", fillcolor='darkseagreen2')
    
    steps = len(genotype)

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        op, j = genotype[i]
        if j == 0:
            u = "input"
        else:
            u = str(j - 1)
        v = str(i)
        g.edge(u, v, label=op, fillcolor="gray")

    g.node("output", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "output", fillcolor="gray")

    g.render(filename, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.geno, "genotype")
