import copy

class HiddenLayerParamSetBuilder:

    @staticmethod
    def BuildHiddenLayerSizeSets(baseline_params, start, step, stop):
    
        results = []

        value = start
        while value <= stop:
            params_copy = copy.deepcopy(baseline_params)
            params_copy["hiddenLayers"] = [value]
            results.append(params_copy)
            value += step

        return results
    ###############################################

    @staticmethod
    def BuildHiddenLayerDepthSets(baseline_params, layer_size, start_layers, stop_layers):
    
        results = []

        depth = start_layers
        while depth <= stop_layers:
            params_copy = copy.deepcopy(baseline_params)
            params_copy["hiddenLayers"] = [layer_size] * depth
            results.append(params_copy)
            depth += 1

        return results
    ###############################################

    @staticmethod
    def BuildHiddenLayerFunnelSets(baseline_params, start_size, step, max_layers):
    
        results = []

        current = start_size
        layers = []

        while len(layers) < max_layers:
            layers.append(current)
            params_copy = copy.deepcopy(baseline_params)
            params_copy["hiddenLayers"] = layers.copy()
            results.append(params_copy)
            current = max(current - step, step)

        return results
    ###############################################
