

def normalize_and_pad_traces2(all_data):
    if not all_data:
        return np.array([])  # Return an empty array if no data

    all_traces = [trace for data in all_data for trace in data[0] if trace.size > 0]
    if not all_traces:
        return np.array([])  # Return an empty array if no valid traces

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, _ in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.extend(padded_traces)  # Flatten the batch for uniform shape

    return np.array(padded_data, dtype='float32')




def normalize_and_pad_traces8(all_data, max_len):
    all_traces = [trace for data in all_data for trace in data[0]]
    if not all_traces:
        return np.empty((0, max_len, 2))  # Adjust dimensions as needed

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    padded_traces = []
    for traces in all_traces:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces.append(pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32'))

    return np.stack(padded_traces)







def normalize_and_pad_tracesb(all_data):
    # Normalize each stroke sequence and determine the global max length
    all_traces = []
    for data in all_data:
        traces = data[0]
        for trace in traces:
            all_traces.append(trace)
    
    if not all_traces:
        return np.empty((0, 0, 0))  # Return an empty 3D array if no traces are present
    
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    max_len = max(len(trace) for trace in all_traces)

    # Pad each trace and normalize
    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))

    # Combine all traces into a single NumPy array
    traces_array = np.vstack([traces for traces, _ in padded_data])
    return traces_array





def normalize_and_pad_traces4(all_data):
    # Flatten all traces to find the global minimum and maximum for normalization
    all_traces = [trace for data in all_data for trace in data[0] if trace.size > 0]
    if not all_traces:
        return np.array([], dtype=float)

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Determine the maximum length of any trace sequence for uniform padding
    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        # Pad sequences to the maximum length
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))

    return padded_data


def read_folder_of_inkml(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.inkml'):
            file_path = os.path.join(folder_path, filename)
            strokes, label = parse_inkml(file_path)
            data.append((strokes, label))
    return data






def normalize_and_pad_traces33(all_data):
    # Normalize each stroke sequence and determine the global max length
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_len = max(len(trace) for trace in all_traces)

    # Pad each trace and stack them into a single NumPy array
    padded_data = []
    for traces, _ in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.extend(padded_traces)  # Extend, not append, to flatten the list

    return np.stack(padded_data)  
    


def normalize_and_pad_traces(all_data):
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))
    return padded_data




def normalize_and_pad_traces_old(all_data):
    # Normalize each stroke sequence and pad data to the maximum sequence length in the batch
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]  # flatten all traces
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Determine the maximum length of any trace sequence for uniform padding
    max_len = max(len(trace) for traces in all_data for trace in traces[0])

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))
    return padded_data