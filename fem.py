import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from scipy.special import legendre

class Geom:
    def __init__(self, domain, dim=1):
        self.xL = domain[0]
        self.xR = domain[1]

class Node:
    def __init__(self, x, node_tag):
        self.x = x
        self.node_tag = node_tag

class MaterialProperties:
    def __init__(self, k=1):
        self.k = k

class Element:
    def __init__(self, nodes, p, elm_tag, material_properties):
        self.nodes   = nodes
        self.elm_tag = elm_tag
        self.p       = p
        self.n_GP    = 2*p + 1
        self.n_GP    = 30
        self.xi, self.w = self.get_legendre()
        self.basis_functions = self.lagrange_basis_functions()
        self.basis_functions_derivative = self.lagrange_basis_derivatives()
        self.h = self.get_h()
        self.material_properties = material_properties

    def get_h(self):
        xmin = np.min( np.array( [self.nodes[i].x for i in range(len(self.nodes))] ) )
        xmax = np.max( np.array( [self.nodes[i].x for i in range(len(self.nodes))] ) )
        h = xmax - xmin
        return h
        
    def get_legendre(self):
        """
        Get abcissae and weights from Gauss-Legende quadrature.
        """
        xi, w = np.polynomial.legendre.leggauss(self.n_GP)
        return xi,w


    def local_to_global(self, xi):
        """
        Get global coordinate value x based on the change of variable xi = a*x + b
        """
        xmin = np.min( np.array( [self.nodes[i].x for i in range(len(self.nodes))] ) )
        xmax = np.max( np.array( [self.nodes[i].x for i in range(len(self.nodes))] ) )
        X = ( xi + (xmax+xmin)/(xmax-xmin) ) * (xmax-xmin)/2
        return X


    def lagrange_basis_functions(self):
        # Generate nodes over the interval [-1, 1]
        nodes = np.array([self.nodes[k].x for k in range(len(self.nodes))])
        nodes = (nodes - np.min(nodes)) / (np.max(nodes) - np.min(nodes)) * 2 - 1

        # Store basis functions as polynomials
        basis_functions = []
        for i in range(self.p + 1):
            y = np.zeros(self.p + 1)
            y[i] = 1
            poly = lagrange(nodes, y)
            basis_functions.append(poly)
        return basis_functions#, nodes

    def lagrange_basis_derivatives(self):
        # Get the basis functions
        basis_functions = self.lagrange_basis_functions()
        
        # Calculate derivatives of basis functions
        basis_function_derivatives = []
        for i, poly in enumerate(basis_functions):
            # Compute the derivative of the polynomial
            derivative = poly.deriv()
            basis_function_derivatives.append(derivative)
        
        return basis_function_derivatives

    def K_e(self):
        """
        Returns K matrix of the element
        """
        K_e = np.zeros([self.p+1, self.p+1])
        for k in range(len(self.xi)):
            for i in range(self.p+1):
                for j in range(self.p+1):
                    K_e[i,j] -= (2/self.h) * self.w[k] * self.basis_functions_derivative[i](self.xi[k]) * self.basis_functions_derivative[j](self.xi[k])
        return K_e

    def M_e(self):
        """
        Returns M matrix of the element
        """
        M_e = np.zeros([self.p+1, self.p+1])
        for k in range(len(self.xi)):
            for i in range(self.p+1):
                for j in range(self.p+1):
                    M_e[i,j] += self.material_properties.k**2 * (self.h/2) * self.w[k] * self.basis_functions[i](self.xi[k]) * self.basis_functions[j](self.xi[k])
        return M_e


class BoundaryCondition:
    def __init__(self, bc_type, bc_val, nodes):
        self.bc_type = bc_type
        self.bc_val  = bc_val
        self.nodes    = nodes


def compute_gll_nodes(n):
    if n < 2:
        raise ValueError("Number of GLL nodes must be at least 2.")
    P_n_minus_1 = legendre(n - 1)
    P_n_minus_1_derivative = np.polyder(P_n_minus_1)
    roots = np.roots(P_n_minus_1_derivative)
    gll_nodes = np.sort(np.concatenate(([-1], roots, [1])))
    return gll_nodes

def generate_mesh(geom, N=1, p=1, k=1, type='gll'):
    # Define nodes
    nodes = []
    if type == 'gll':
        x_gll = compute_gll_nodes(p+1)
        h = (geom.xR - geom.xL)/N
        l = 0
        for i in range(N):
            for j in range(p+1 - 1):
                node = Node(i*h + (x_gll[j]+1)/2*h, l)
                nodes.append(node)
                l+=1
        node = Node(geom.xR, l)
        nodes.append(node)
    elif type == 'eq':
        x_eq  = np.linspace(geom.xL, geom.xR, N*p+1)
        for i in range(N*p + 1):
            node_i = Node(x_eq[i], i)
            nodes.append(node_i)



    # Define elements
    elements = []
    for i in range(N):
        xmin = np.min( [nodes[k].x for k in range(i*p, i*p+p+1)] )
        xmax = np.max( [nodes[k].x for k in range(i*p, i*p+p+1)] )
        x_barycentre = 0.5 * (xmin + xmax)
        material_properties = MaterialProperties(k(x_barycentre)) 
        print(f'k={k(x_barycentre)}')

        elm_i = Element([nodes[k] for k in range(i*p, i*p+p+1)], p, i, material_properties)
        elements.append(elm_i)

    return elements, nodes


def generate_mesh_old(geom, N=1, p=1):
    n_nodes = N*p + 1
    x = np.linspace(geom.xL, geom.xR, n_nodes)

    # Define nodes
    nodes    = []
    for i in range(n_nodes):
        node_i = Node(x[i], i)
        nodes.append(node_i)

    # Define elements
    elements = []
    for i in range(N):
        elm_i = Element([nodes[k] for k in range(i*p, i*p+p+1)], p, i)
        elements.append(elm_i)

    return elements, nodes

def assemble_K(elements, nodes, mapping):
    K = np.zeros([len(nodes), len(nodes)])
    for elm in elements:
        K[ np.ix_( [elm.nodes[k].node_tag for k in range(len(elm.nodes))] , [elm.nodes[k].node_tag for k in range(len(elm.nodes))] ) ] += elm.K_e()
    return K

def assemble_M(elements, nodes, mapping):
    M = np.zeros([len(nodes), len(nodes)])
    for elm in elements:
        M[ np.ix_( [elm.nodes[k].node_tag for k in range(len(elm.nodes))] , [elm.nodes[k].node_tag for k in range(len(elm.nodes))] ) ] += elm.M_e()
    return M

def assemble_f(fun, elements, nodes, mapping):

    def f_e(fun, elm):
        f_e = np.zeros(elm.p+1)
        for k in range(len(elm.xi)):
            for i in range(elm.p+1):
                f_e[i] += (elm.h/2) * elm.w[k] * elm.basis_functions[i](elm.xi[k]) * fun(elm.local_to_global(elm.xi[k]))
        return f_e

    f = np.zeros(len(nodes))
    for elm in elements:
        f[ np.ix_( [elm.nodes[k].node_tag for k in range(len(elm.nodes))] ) ] += f_e(fun, elm)
    return f

def print_geom_properties(geom, elements, nodes):
    print('Geometry:')
    print(f'x_L = {geom.xL}, x_R = {geom.xR}')
    print('elements:')
    for elm in elements:
        print(f'element tag = {elm.elm_tag}')
        for node in elm.nodes:
            print(f'> node tag = {node.node_tag}, x = {node.x}')
        print(f'Elemental mass matrix M_e :')
        print(elm.M_e())
        print(f'Elemental stiffness matrix K_e :')
        print(elm.K_e())

def apply_bc(A,f,bc, tag2index):
    for node in sorted(bc.nodes, key=lambda node: node.node_tag, reverse=True):
        if bc.bc_type == 'Dirichlet':
            for i in range(len(f)):
                f[i] -= A[tag2index[node.node_tag], i] * bc.bc_val
            A = np.delete(A, tag2index[node.node_tag], axis=0)
            A = np.delete(A, tag2index[node.node_tag], axis=1)
            f = np.delete(f, tag2index[node.node_tag])

        elif bc.bc_type == 'Neumann':
            f[tag2index[node.node_tag]] -= bc.bc_val
        else:
            print(f'Error : bc_type {bc.bc_type} is not supported')
    return A, f


def tag_to_index(nodes):
    mapping = {}
    for i in range(len(nodes)):
        mapping[nodes[i].node_tag] = i
    return mapping

def reverse_map(mapping):
    rev_mapping = {}
    for key, value in mapping.items():
        rev_mapping[value] = key
    return rev_mapping


def main():
    # Define geometry and mesh
    geom = Geom(domain=[0,1], dim=1)
    k = 10*np.pi

    elements, nodes = generate_mesh(geom, N=1, p=2, k=k)
    tag2index = tag_to_index(nodes)

    # Define rhs
    source = lambda x : 1*np.sin(2*np.pi*x)

    # Assemble K and M
    K = assemble_K(elements, nodes, tag2index)
    M = assemble_M(elements, nodes, tag2index)
    f = assemble_f(source, elements, nodes, tag2index)

    A = K + M


    # Apply BCs
    bc_D = BoundaryCondition('Dirichlet', 0, [nodes[0]])
    bc_N = BoundaryCondition('Neumann',   0, [nodes[-1]])
    A, f = apply_bc(A, f, bc_N, tag2index)
    A, f = apply_bc(A, f, bc_D, tag2index)

    # Solve system
    sol = np.matmul(np.linalg.inv(A) , f)

    # Encode solution
    for node in bc_D.nodes:
        del tag2index[node.node_tag]
        for key in tag2index:
            if key > node.node_tag:
                tag2index[key] -= 1
    u = np.zeros(len(nodes))
    x = np.zeros(len(nodes))
    for k in range(len(nodes)):
        x[k] = nodes[k].x
        if nodes[k].node_tag in {bc_node.node_tag for bc_node in bc_D.nodes}:
            u[k] = bc_D.bc_val
        else:
            u[k] = sol[tag2index[nodes[k].node_tag]]

    # Plot results
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(x, u, lw=1, color='b')
    ax.grid(True)
    ax.tick_params(axis='both', direction='in', length=6, top=True, right=True, labelbottom=True)
    plt.show()

if __name__ == '__main__':
    main()
