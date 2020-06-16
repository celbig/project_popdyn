import numpy as np
import numpy.polynomial.polynomial as poly
import plotly.graph_objects as go
import time
from scipy.integrate import ode
from plotly.subplots import make_subplots
from kuznetsov_model.helper_functions import * 


class ODE_Model:
    """
    A class that implements the ode model of kuznetsov (1994)
    The class hold the model parameter and can:
        -- generate the nucline graph
        -- Compute steady states and analyse them
        -- generate the phase graph (with manifolds and trajectories)
        -- simulate the realization of a stochastic version of the models

    credits:
        written by Célestin BIGARRÉ for the cellular population dynamics course
        (M2 Maths en action UCBL Lyon 1)
        2020

    license: CC BY SA
    """
    def __init__(self,
                 sigma = 0.1181,
                 rho = 1.131,
                 eta= 20.19,
                 mu = 0.00311,
                 delta = 0.3743,
                 beta = 2.0*10**(-3),
                 alpha = 1.636,
                 T0 = 10**6,
                 E0 = 10**6):
        """
        Constructor for the class
        """
        # Model parameters
        self.sigma = sigma
        self.rho = rho
        self.eta = eta
        self.mu = mu
        self.delta = delta
        self.beta = beta
        self.alpha = alpha
        self.T0 = T0
        self.E0 = E0
        self.T_max = 200

        # class variables
        self.polynomial_coeffs = None
        self.nullcline_x = None
        self.nullcline_y = None
        self.nullcline_graph = None
        self.phase_graph = None
        self.Jacobian = None
        self.ode_function = None
        self.legend_traj = None

        # Initialization
        self.__initialize()

    def __initialize(self):
        """
        Helper function for initializing the class
        """

        #initialize model funtions
        self.nullcline_x = lambda y : self.sigma / (self.delta + self.mu * y - y * self.rho / (self.eta + y))
        self.nullcline_y = lambda y : self.alpha * (1 - self.beta * y)
        self.ode_function = lambda t, x : np.array([
            self.sigma + self.rho * x[0] *x[1] /(self.eta + x[1]) - self.mu * x[0] *x[1] - self.delta * x[0],
            self.alpha * x[1] * (1 - self.beta * x[1]) - x[0] *x[1]
        ])

        self.__get_polynomial()
        self.__get_Jacobian()

        #delete phase graph
        self.phase_graph = None


    def set_param(self,
                 sigma = None,
                 rho = None,
                 eta= None,
                 mu = None,
                 delta = None,
                 beta = None,
                 alpha = None,
                 T0 = None,
                 E0 = None):
        """
        Change one or more parameter of the model instance
        """
        if sigma != None:
            self.sigma = sigma
        if rho != None:
            self.rho = rho
        if eta != None:
            self.eta = eta
        if mu != None:
            self.mu = mu
        if delta != None:
            self.delta = delta
        if beta != None:
            self.beta = beta
        if alpha != None:
            self.alpha = alpha
        if T0 != None:
            self.T0 = T0
        if E0 != None:
            self.E0 = E0

        # apply changes
        self.__initialize()

    def __get_polynomial(self):
        """
        compute coeffs of the steady-state polynomial (used for initialization)
        """
        C0 = self.eta * ((self.sigma / self.alpha) - self.delta)
        C1 = self.rho + self.beta * self.eta * self.delta + (self.sigma / self.alpha) - self.delta - self.mu * self.eta
        C2 = self.beta *(self.delta + self.eta * self.mu - self.rho) - self.mu
        C3 = self.beta * self.mu
        self.polynomial_coeffs = np.array([C0, C1, C2, C3])

    def get_nullcline_graph(self,
        y_range,
        y_step,
        x_log = False,
        y_log = False,
        only_positive_ss = False
    ):
        """
        Generates the nucline graph with approximate intersections
        INPUTS:
            y_range = [y_min, y_max] -> y range over which to compute Nullcline
            y_step -> step for y grid
            x_log -> set x_axis in log form
            y_log -> set y axis in log form
            only_positive_ss -> if True only positive steady states (of bological interest will be ploted)
        OUTPUTS:
            the nucline graph as a plotly Figure object
        """
        # compute the nuclines
        y = np.arange(y_range[0], y_range[1], y_step)
        x_1= self.nullcline_y(y)
        x_2= self.nullcline_x(y)

        if only_positive_ss :
            y_pos = y[(x_1 > 0) & (x_2 > 0)]
            x_1pos = x_1[(x_1 > 0) & (x_2 > 0)]
            x_2pos = x_2[(x_1 > 0) & (x_2 > 0)]
        else :
            y_pos = y
            x_1pos = x_1
            x_2pos = x_2

        # generate the figure
        self.nullcline_graph = go.Figure(
            data=[go.Scatter(x = x_1pos,
                             y = y_pos,
                             mode="lines",
                             line=go.scatter.Line(color="blue"),
                             name = "x = g(x)")
                 ],
            layout=go.Layout(
                title=go.layout.Title(text="Nullclines, beta = " + str(self.beta) + ", alpha = " + str(self.alpha))
            )
        )
        self.nullcline_graph.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        self.nullcline_graph.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

        if x_log:
            self.nullcline_graph.update_xaxes(type="log")
        if y_log:
            self.nullcline_graph.update_yaxes(type="log")

        # compute forbidden values of the function
        y_forbidden = []
        if -self.eta >= y_range[0] and -self.eta <= y_range[1]:
            y_forbidden += [-self.eta]

        D = (self.delta + self.eta * self.mu - self.rho) ** 2 - 4 * self.mu * self.delta * self.eta
        if D > 0:
            y1 = (self.rho-self.eta * self.mu - self.delta - np.sqrt(D)) / (2 * self.mu)
            y2 = (self.rho-self.eta * self.mu - self.delta + np.sqrt(D)) / (2 * self.mu)
            if y1 >= y_range[0] and y1 <= y_range[1] :
                y_forbidden += [y1]
            if y2 >= y_range[0] and y2 <= y_range[1] :
                y_forbidden += [y2]
        if D == 0:
            y1 = (self.rho-self.eta * self.mu * self.delta) / (2 * self.mu)
            if y1 >= y_range[0] and y1 <= y_range[1] :
                y_forbidden += [y1]
        y_forbidden += [y_range[1]]
        y_forbidden = np.sort(np.array(y_forbidden))

        y_intersect = np.array([])

        # plot piecewise between forbidden values
        for i, y_f in enumerate(y_forbidden):
            if i == 0:
                index_local = y_pos < y_f
            elif i >= len(y_forbidden):
                index_local = y_pos > y_f
            else :
                index_local = (y_pos < y_f) & (y_pos > y_forbidden[i-1])

            y_local = y_pos[index_local]
            x_local = x_2pos[index_local]
            self.nullcline_graph.add_trace(go.Scatter(x = x_local,
                                                     y = y_local,
                                                     mode="lines",
                                                     line=go.scatter.Line(color="red"),
                                                     name = "x = f(y)",
                                                     showlegend= (i == 0) ))
            # determine y of approximate intersection points between nuclines
            y_intersect = np.append(y_intersect, function_intersect(y_local, x_local, x_1pos[index_local]))

        # compute x for intersection points
        x_intersect = np.mean([self.nullcline_y(y_intersect), self.nullcline_x(y_intersect)], axis = 0)
        y_intersect = np.append(y_intersect , [0])
        x_intersect = np.append(x_intersect , [self.nullcline_x(0)])

        y_intersect_pos = y_intersect[(x_intersect >=0) & (y_intersect >=0)]
        x_intersect_pos = x_intersect[(x_intersect >=0) & (y_intersect >=0)]

        x_intersect_non_pos=x_intersect[(x_intersect < 0) | (y_intersect  < 0)]
        y_intersect_non_pos=y_intersect[(x_intersect < 0) | (y_intersect < 0)]

        #plot interection points
        self.nullcline_graph.add_trace(
            go.Scatter(
                x=x_intersect_pos,
                y=y_intersect_pos,
                name='Positive Steady-states : ' + str(len(x_intersect_pos)),
                mode='markers',
                fillcolor='orange',
                marker_line_width=2,
                marker_size=10
            ))
        if not only_positive_ss:
            self.nullcline_graph.add_trace(
                go.Scatter(
                    x=x_intersect_non_pos,
                    y=y_intersect_non_pos,
                    name='Non Positive Steady-states : ' + str(len(x_intersect_non_pos)),
                    mode='markers',
                    fillcolor='gray',
                    marker_line_width=2,
                    marker_size=10
                ))

        return self.nullcline_graph

    def get_steady_states(self, positives_only = False):
        """
            return the list of steady states for the model
            INPUTS :
                positives_only -> if True, only returns positives steady_states
            OUTPUTS :
                [ss] -> a list of steady states where ss is a dict with keys :
                    'x' -> x position
                    'y' -> y position²
                    'eig_vals' -> a list of the jacobian eigen values
                    'eig_vects' -> a matrix such that eig_vects[:,k] is the kth eigen vector of the jacobian
                    'name' -> a letter unique to each steady state (ordered by y position)
        """

        letters = ['A', 'B', 'C', 'D', 'E']

        #compute the polynomial roots
        roots = poly.polyroots(self.polynomial_coeffs)
        roots = np.real(roots[(np.isreal(roots))])
        if positives_only:
            roots = roots[roots >= 0]

        # analyse the "y = 0" steady state
        steady_states = [self.__analyse_steady_state(self.nullcline_x(0), 0, letters[0])]

        # analyse each steady state
        i = 1
        for ss_y in roots:
            ss_x = self.nullcline_x(ss_y)
            if ss_x >=0 or (not positives_only):
                steady_states += [self.__analyse_steady_state(ss_x, ss_y, letters[i])]
            i += 1

        return steady_states

    def __analyse_steady_state(self, x, y, letter):
        """
        given a stedy state, analyse it (private function, not to be called outside the class)
        """
        #compute the jacobian at the steady_state
        ss_jacobian = self.Jacobian(x, y)

        #compute eigen values and vectors
        eig_vals, eig_vects = np.linalg.eig(ss_jacobian)
        real_ev = np.real(eig_vals)

        #general info for the steady state
        ss = {'x': x,
             'y' : y,
             'eig_vals': eig_vals,
             'eig_vect': eig_vects,
             'name' : letter}

        # stability analysis
        if any(real_ev == 0):
            ss['stability'] = 'Unknown'
        elif all(real_ev > 0):
            ss['stability'] = 'Unstable'
        elif all(real_ev < 0):
            ss['stability'] = 'Stable'
        else :
            ss['stability'] = 'Saddle'

        return ss

    #Compute the Jacobian:
    def __get_Jacobian(self):
        """
        generate  the jacobian function from the model parameters
        private function for initialization only
        OUTPUT:
            return a function (x, y) -> J(x,y)
        """
        self.Jacobian = lambda x, y : np.array([
            [self.rho * y /(self.eta + y) - self.mu * y - self.delta,
            (self.rho * x * (self.eta + y) - self.rho * x * y)/(self.eta + y) ** 2 - self.mu * x],
            [-y,
            self.alpha * (1 - self.beta * y) - self.beta * self.alpha * y - x]
        ])

    def get_phase_graph(self):
        """
        return the model phase_graph
        """

        if self.phase_graph == None:
            self.reset_phase_graph()
        return self.phase_graph


    def reset_phase_graph(self,
        select_unstable_manifold = None,
        select_stable_manifold = None
    ):
        """
        (re-)initialize  the model phase graph
        erase all previous computed trajectories
        INPUTS:
            select_unstable_manifold : a list of saddle steady states names ('A'..'E') for which unstable manifold sould be drawn, if None unstable manifold for all Saddle Steady states are drawn. In all cases, manifolds are drawn only for saddle steady states.
            select_unstable_manifold -> idem for stable manifolds
        OUTPUTS:
            phase graph as a plotly Figure object
        """
        self.legend_traj = True
        stables_ss = {}
        unstables_ss = {}
        saddle_ss = {}
        stable_manifolds = []
        unstable_manifolds = []

        # Get steady states
        steady_states = self.get_steady_states(positives_only = True)

        # Class steady states by stability
        for ss in steady_states:
            if ss['stability'] == 'Stable':
                stables_ss = add_ss_to_list(ss, stables_ss)
            elif ss['stability'] == 'Unstable':
                unstables_ss = add_ss_to_list(ss, unstables_ss)
            elif ss['stability'] == 'Saddle':
                saddle_ss = add_ss_to_list(ss, saddle_ss)
                # Get manifolds
                if (select_stable_manifold == None) or (ss['name'] in select_stable_manifold):
                    stable_manifolds += self.__get_stable_manifold(ss)
                if (select_unstable_manifold == None) or (ss['name'] in select_unstable_manifold):
                    unstable_manifolds += self.__get_unstable_manifold(ss)

        # Plot
        self.phase_graph = go.Figure(data = [],
                                    layout=go.Layout(
                title=go.layout.Title(text="Phase portrait")
            ))
        if len(stables_ss) != 0:
            self.phase_graph.add_trace(go.Scatter(x = stables_ss['x'],
                                 y = stables_ss['y'],
                                 text = stables_ss['name'],
                                 mode="markers+text",
                                 marker_size = 10,
                                 marker_opacity = 1,
                                 textposition="top center",
                                 name = "stable steady states"))
        if len(unstables_ss) != 0:
            self.phase_graph.add_trace(go.Scatter(x = unstables_ss['x'],
                                 y = unstables_ss['y'],
                                 text = unstables_ss['name'],
                                 mode="markers+text",
                                 marker_size = 10,
                                 marker_opacity = 1,
                                 textposition="top center",
                                 name = "unstable steady states"))
        if len(saddle_ss) != 0:
            self.phase_graph.add_trace(go.Scatter(x = saddle_ss['x'],
                                 y = saddle_ss['y'],
                                 text = saddle_ss['name'],
                                 mode="markers+text",
                                 marker_size = 10,
                                 marker_opacity = 1,
                                 textposition="top center",
                                 name = "saddle steady states"))
        if len(stable_manifolds) != 0:
            legend = True
            for manifold in stable_manifolds:
                self.phase_graph.add_trace(go.Scatter(x = manifold[0,:],
                                 y = manifold[1,:],
                                 mode="lines",
                                 line_color ="crimson",
                                 name = "stable manifolds",
                                 showlegend=legend))
                legend = False

        if len(unstable_manifolds) != 0:
            legend = True
            for manifold in unstable_manifolds:
                self.phase_graph.add_trace(go.Scatter(x = manifold[0,:],
                                 y = manifold[1,:],
                                 mode="lines",
                                 line_color ="cadetblue",
                                 line_dash = 'dash',
                                 name = "unstable manifolds",
                                 showlegend=legend))
                legend = False

        return self.phase_graph

    def __get_unstable_manifold(self, steady_state):
        """
        Compute unstable manifolds for a steady state
        INPUTS:
            steady_state : a steady state in a dict format
        """
        manifolds =[]
        ss_x = steady_state['x']
        ss_y = steady_state['y']
        ss_pos = np.array([ss_x, ss_y])

        #ode solver
        epsilon = 0.05
        dt = 0.1
        ode_solver = ode(self.ode_function).set_integrator('dopri5')

        #compute manifold for each non-negative eigen value
        for i in range(len(steady_state['eig_vals'])):
            eig_val = steady_state['eig_vals'][i]
            eig_vect = steady_state['eig_vect'][:, i]
            if eig_val > 0 :
                t = np.arange(0, self.T_max, dt)
                half_manifold = np.empty((2, len(t)))
                half_manifold[:,0] = [ss_x, ss_y]

                #compute the half manifod for +eigen_vect
                ode_solver.set_initial_value(ss_pos + epsilon * eig_vect, 0)
                i = 1
                while ode_solver.successful() and i < len (t):
                    half_manifold[:,i] = ode_solver.integrate(t[i])
                    i += 1
                if i != len(t):
                    half_manifold = half_manifold[:, range(i)]
                manifolds += [half_manifold]


                half_manifold = np.empty((2, len(t)))
                half_manifold[:,0] = [ss_x, ss_y]

                #compute the half manifod for -eigen_vect
                ode_solver.set_initial_value(ss_pos - epsilon * eig_vect, 0)
                i = 1
                while ode_solver.successful() and i < len (t):
                    half_manifold[:,i] = ode_solver.integrate(t[i])
                    i += 1
                if i != len(t):
                    half_manifold = half_manifold[:, range(i)]
                manifolds += [half_manifold]

        return manifolds

    def __get_stable_manifold(self, steady_state):
        """
        Compute stable manifolds for a steady state
        INPUTS:
            steady_state : a steady state in a dict format
        """
        manifolds =[]
        ss_x = steady_state['x']
        ss_y = steady_state['y']
        ss_pos = np.array([ss_x, ss_y])
        # we integrate in negative times
        f = lambda t, x : - self.ode_function(t,x)

        #ode solver
        epsilon = 0.05
        dt = 0.1
        ode_solver = ode(f).set_integrator('dopri5')

        #compute manifold for each non-positive eigen value
        for i in range(len(steady_state['eig_vals'])):
            eig_val = steady_state['eig_vals'][i]
            eig_vect = steady_state['eig_vect'][:, i]
            if eig_val < 0 :
                t = np.arange(0, self.T_max, dt)
                half_manifold = np.empty((2, len(t)))
                half_manifold[:,0] = [ss_x, ss_y]

                #compute the half manifod for +eigen_vect
                ode_solver.set_initial_value(ss_pos + epsilon * eig_vect, 0)
                i = 1
                while ode_solver.successful() and i < len (t):
                    half_manifold[:,i] = ode_solver.integrate(t[i])
                    i += 1
                if i != len(t):
                    half_manifold = half_manifold[:, range(i)]
                manifolds += [half_manifold]


                half_manifold = np.empty((2, len(t)))
                half_manifold[:,0] = [ss_x, ss_y]

                #compute the half manifod for -eigen_vect
                ode_solver.set_initial_value(ss_pos -epsilon * eig_vect, 0)
                i = 1
                while ode_solver.successful() and i < len (t):
                    half_manifold[:,i] = ode_solver.integrate(t[i])
                    i += 1
                if i != len(t):
                    half_manifold = half_manifold[:, range(i)]
                manifolds += [half_manifold]

        return manifolds


    def add_trajectory_to_phase_graph(self,
        x,
        y,
        T_max = 400,
        dt = 0.1
    ):
        """
        Update the phase graph with a new trajectory
        INPUTS:
            x -> x position of the starting point
            y -> y position of the starting point
            T_max -> max time for the trajectory computation
            d_t -> trajectory time step
        """
        t = np.arange(0, T_max, dt)
        traj = np.empty((2, len(t)))
        traj[:,0] = [x, y]
        i = 1

        # ODE solver
        ode_solver = ode(self.ode_function).set_integrator('dopri5')
        ode_solver.set_initial_value([x, y], 0)

        # Compute trajectory
        while ode_solver.successful() and i < len (t):
            traj[:,i] = ode_solver.integrate(t[i])
            i += 1
        if i != len(t):
            traj = traj[:, range(i)]

        # Verify that the phase graph exists
        if self.phase_graph == None:
            self.reset_phase_graph()

        # plot
        self.phase_graph.add_trace(
            go.Scatter(
                x = [x],
                y = [y],
                mode = "markers",
                marker_symbol = "x",
                marker_color = "black",
                name = "Trajectories starting points",
                showlegend = self.legend_traj
        ))
        self.phase_graph.add_trace(
            go.Scatter(
                x = traj[0,:],
                y = traj[1,:],
                mode = "lines",
                line_color = "black",
                line_dash = "dot",
                line_width=0.75,
                name =  "Trajectories",
                showlegend = self.legend_traj
        ))

        #Not to plot trajectory legend several times
        self.legend_traj = False


    def simul_stoch_model(self,
        Nx0,
        Ny0,
        V,
        Tmax,
        dtmin = 0.1,
        generate_graph = False, 
        max_compute_time = None
    ):
        """
        Simulate the stochastic version of the model
        INPUTS:
            Nx0 -> initial number of Effecto Cells
            Ny0 -> initial number of tumor Cells
            V -> Volume for the computation
            dt_min -> Limit parameter to avoid too small simulation steps
            generate_graph -> if True, also return a graph
        """
        # Array declarations
        n_array = int(np.ceil(Tmax))
        t = np.empty((n_array,))
        dt_list = np.zeros((n_array,))
        process = np.empty((2,n_array))

        # Initialization
        i = 0
        process[:,0] = [Nx0, Ny0]
        t[0] = 0
        cases = [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
        ]
        cases_indexes = [0, 0, 1, 1, 2, 3, 3]
        begin_timer = time.time()

        if not max_compute_time:
            max_compute_time = np.Inf

        # Simulation
        while t[i] < Tmax  and process[1, i] > 0 and process[0, i] > 0  and time.time() - begin_timer < 30 :
            x = process[0, i]
            y = process[1, i]

            # Extend array size if necessary
            if i >= n_array-1:
                process = np.concatenate([process, np.empty((2,n_array))], axis = 1)
                dt_list = np.concatenate([dt_list, np.zeros((n_array,))])
                t = np.concatenate([t, np.empty((n_array,))])
                n_array = t.size

            # Compute transition probabiliies
            prob = np.array([
                V  * self.sigma,
                self.rho * x * y /(V * self.eta + y),
                self.mu * x * y / V,
                self.delta *x,
                self.alpha * y,
                self.alpha * y * self.beta * y / V,
                x * y / V
            ])
            #prob = prob / np.sum(prob)

            # Process update
            k = np.random.choice(cases_indexes, p = prob / np.sum(prob) )
            t[i + 1] = t[i] + dt_list[i]
            process[:, i + 1]  = cases[k] + process[:, i]

            # Time update
            dt_list[i] = np.random.exponential(1 / np.sum(prob))
            t[i + 1] = t[i] + dt_list[i]

            i += 1

        # resize array
        process = process[:, 0:i]
        t = t[0:i]

        if not generate_graph:
            return (t, process)

        # plot
        graph = make_subplots(rows=2, cols=1,
                             subplot_titles=[
                                 "Nx process",
                                 "Ny process"
                             ])
        graph.update_layout(title_text="$\\text{Stochastic process}$")

        graph.add_trace(go.Scatter(
            x = t,
            y = process[0,:],
            mode = "lines",
            name = "Nx"
        ), row = 1, col = 1)
        graph.update_xaxes(title_text="t", row=1, col=1)
        graph.update_yaxes(title_text="Nx", row=1, col=1)
        graph.add_trace(go.Scatter(
            x = t,
            y = process[1,:],
            mode = "lines",
            name = "Ny"
        ), row = 2, col = 1)
        graph.update_xaxes(title_text="t", row=2, col=1)
        graph.update_yaxes(title_text="Nx", row=2, col=1)
        return (t, process, graph)
