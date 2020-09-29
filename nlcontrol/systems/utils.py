import csv
import sys
from os import path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def write_simulation_result_to_csv(simulation_result, file_name=None):
    """
    Write the results of a SimulationResult object (see simupy.BlockDiagram.simulate) to a csv file. This object type is also returned by a SystemBase's simulation function.

    Parameters
    -----------
    simulation_result : SimulationResult object or list
        Results of a simulation packaged as Simupy's SimulationResult object or a list which includes the time, input, state, and output vector in this order.
    file_name : string
        The filename of the newly created csv file. Defaults to a timestamp.


    Examples
    ---------
    * Simulate a SystemBase object called 'sys' and store the results:
        >>> t, x, y, u, res = sys.simulation(1)
        >>> write_simulation_result_to_csv(res, file_name='use_simulation_result_object')
        >>> write_simulation_result_to_csv([t, u, x, y], file_name='use_separate_vectors')

    """
    if hasattr(sys.modules['__main__'], '__file__'):
        abs_path = path.dirname(path.realpath(sys.modules['__main__'].__file__))

        if file_name is None:
            now = datetime.now()
            now_formatted = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
            file_name = now_formatted + ".csv"
        elif ".csv" not in file_name:
            file_name = file_name + ".csv"
        print('Writing to: ', file_name)

        file_path = path.join(abs_path, file_name)
        if (isinstance(simulation_result, list)):
            t = simulation_result[0]
            u = simulation_result[1]
            x = simulation_result[2]
            y = simulation_result[3]
            e = None
        else:
            t = simulation_result.t
            u = None
            x = simulation_result.x
            y = simulation_result.y # contains u vectors
            e = simulation_result.e

        with open(file_path, "w", newline='') as f:
            if e is None:
                fieldnames = ['t', 'x', 'y', 'u']
            else:
                fieldnames = ['t', 'x', 'y', 'e']
            csv_writer = csv.writer(f, delimiter=';')
            csv_writer.writerow(fieldnames)
            try:
                for i in range(len(t)):
                    row_temp = [t[i],  x[i, :], y[i, :]]
                    if e is None:
                        row_temp.append(u[i, :]) 
                    else:
                        row_temp.append(e[i, :])
                    csv_writer.writerow(row_temp)
            except csv.Error as e:
                error_text = 'file {}, line {}: {}'.format(file_name, i, e)
                raise SystemExit(error_text)

    else:
        error_text = "[nlcontrol.systems.utils] The function 'write_simulation_result_to_csv' cannot be used in a dynamically created module."
        raise SystemExit(error_text)


def read_simulation_result_from_csv(file_name, plot=False):
    """
    Read a csv file created with `write_simulation_result_to_csv()` containing simulation results. Based on the header it is determined if the results contains input or event vector. There is a possibility to create plot of the data.

    Parameters
    -----------
    file_name : string
        The filename of the csv file, containing the extension.
    plot : boolean, optional
        Create a plot, default: False


    Returns
    --------
    tuple :
        t : numpy array
            The time vector.
        x : numpy array
            The state vectors.
        y : numpy array
            The output vectors. Contains the inputs, when the data contains the event vector.
        u or e : numpy array
            The input vectors or event vectors. See boolean 'contains_u' to know which one.
        contains_u : boolean
            Indicates whether the output contains the input or event vector.


    Examples
    ---------
    * Read and plot a csv file 'results.csv' with an input vector:
        >>> t, x, y, u, contains_u = read_simulation_result_from_csv('results.csv', plot=True)
        >>> print(contains_u)
            True

    * Read and plot a csv file 'results.csv' with an event vector:
        >>> t, x, y, e, contains_u = read_simulation_result_from_csv('results.csv', plot=True)
        >>> print(contains_u)
            False

    """
    def convert_string_reads(row):
        # csv reader returns all data in string type. Convert back to floats.
        return [convert_string_reads(list(row_element.replace("[","").replace("]","").split())) if "[" in row_element else float(row_element) for row_element in row]

    if hasattr(sys.modules['__main__'], '__file__'):
        abs_path = path.dirname(path.realpath(sys.modules['__main__'].__file__))
        file_path = path.join(abs_path, file_name)
        with open(file_path, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter=";")
            t = []
            u = []
            x = []
            y = []
            e = []
            for i, row in enumerate(csv_reader):
                if i == 0:
                    header = row
                else:
                    row = convert_string_reads(row)
                    t.append(row[0])
                    x.append(row[1])
                    y.append(row[2])
                    if 'e' in header:
                        e.append(row[3])
                        u = None
                    else:
                        u.append(row[3])
                        e = None
            t = np.array(t)
            x = np.array(x)
            y = np.array(y)
            
            if 'e' in header:
                e = np.array(e)
                if plot:
                    plt.figure()
                    plt.subplot(121)
                    for i in range(len(x[0])):
                        plt.plot(t, x[:, i], label='x' + str(i))
                    plt.title('states versus time')
                    plt.xlabel('time (s)')
                    plt.legend()
                    plt.subplot(122)
                    for j in range(len(e[0])):
                        plt.plot(t, e[:, j], label='e' + str(j))
                    for k in range(len(y[0])):
                        plt.plot(t, y[:, k], label='y' + str(k))
                    plt.title('events and outputs versus time')
                    plt.xlabel('time (s)')
                    plt.legend()
                    plt.show()
                return t, x, y, e, False
            else:
                u = np.array(u)
                if plot:
                    plt.figure()
                    plt.subplot(121)
                    for i in range(len(x[0])):
                        plt.plot(t, x[:, i], label='x' + str(i))
                    plt.title('states versus time')
                    plt.xlabel('time (s)')
                    plt.legend()
                    plt.subplot(122)
                    for j in range(len(u[0])):
                        plt.plot(t, u[:, j], label='u' + str(j))
                    for k in range(len(y[0])):
                        plt.plot(t, y[:, k], label='y' + str(k))
                    plt.title('inputs and outputs versus time')
                    plt.xlabel('time (s)')
                    plt.legend()
                    plt.show()
                return t, x, y, u, True
    else:
        error_text = "[nlcontrol.systems.utils] The function 'write_simulation_result_to_csv' cannot be used in a dynamically created module."
        raise SystemExit(error_text)