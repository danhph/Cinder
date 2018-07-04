import re
import lief
import hashlib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import logging


class FeatureType(object):
    """Base class that each feature group must implement. """

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, raw_bytes, lief_binary):
        raise NotImplemented

    def process_raw_features(self, raw_obj):
        raise NotImplemented

    def feature_vector(self, raw_bytes, lief_binary):
        return self.process_raw_features(self.raw_features(raw_bytes, lief_binary))


class ByteHistogram(FeatureType):
    """ Byte histogram over the entire binary file """

    name = 'histogram'
    dim = 256

    def raw_features(self, raw_bytes, lief_binary):
        counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        total = counts.sum()
        normalized = counts / total
        return normalized


class ByteEntropyHistogram(FeatureType):
    """ 2d byte/entropy histogram based on Saxe and Berlin, 2015. """

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window

        # x2 because we reduced information by half
        i = np.nonzero(c)[0]
        entropy = np.sum(-p[i] * np.log2(p[i])) * 2

        #  16 bins and max entropy is 8 bits
        entropy_bin = int(entropy * 2)
        if entropy_bin == 16:
            entropy_bin = 15

        return entropy_bin, c

    def raw_features(self, raw_bytes, lief_binary):
        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(raw_bytes, dtype=np.uint8)
        if a.shape[0] < self.window:
            entropy_bin, c = self._entropy_bin_counts(a)
            output[entropy_bin, :] += c
        else:
            # TODO: error occurred when running with Python 32-bit in Windows OS
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            for block in blocks:
                entropy_bin, c = self._entropy_bin_counts(block)
                output[entropy_bin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        total = counts.sum()
        normalized = counts / total
        return normalized


class SectionInfo(FeatureType):
    """ Section names, sizes and entropy. """

    name = 'section'
    # TODO: reduce dimension
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def raw_features(self, raw_bytes, lief_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {
            "entry": entry_section,
            "sections": [
                {
                    'name': s.name,
                    'size': s.size,
                    'entropy': s.entropy,
                    'vsize': s.virtual_size,
                    'props': [str(c).split('.')[-1] for c in s.characteristics_lists]
                } for s in lief_binary.sections]
        }
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            # total number of sections
            len(sections),
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general,
            section_sizes_hashed,
            section_entropy_hashed,
            section_vsize_hashed,
            entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)


class ImportsInfo(FeatureType):
    """ Imported functions """

    name = 'imports'
    dim = 512 + 128

    def raw_features(self, raw_bytes, lief_binary):
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []
            imports[lib.name].extend([entry.name[:10000] for entry in lib.entries])

        return imports

    def process_raw_features(self, raw_obj):
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(128, input_type="string").transform([libraries]).toarray()[0]

        # for examples, "kernel32.dll:CreateFileMappingA" 
        imports = [lib.lower() + ':' + func for lib, func_list in raw_obj.items() for func in func_list]
        imports_hashed = FeatureHasher(512, input_type="string").transform([imports]).toarray()[0]

        return np.hstack([
            libraries_hashed,
            imports_hashed
        ]).astype(np.float32)


class ExportsInfo(FeatureType):
    """ Exported functions. """

    name = 'exports'
    dim = 128

    def raw_features(self, raw_bytes, lief_binary):
        if lief_binary is None:
            return []
        clipped_exports = [export[:10000] for export in lief_binary.exported_functions]
        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)


class GeneralFileInfo(FeatureType):
    """ General information about the file """

    name = 'general'
    dim = 10

    def raw_features(self, raw_bytes, lief_binary):
        if lief_binary is None:
            return {
                'size': len(raw_bytes),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }

        return {
            'size': len(raw_bytes),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signature),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray([
            raw_obj['size'],
            raw_obj['vsize'],
            raw_obj['has_debug'],
            raw_obj['exports'],
            raw_obj['imports'],
            raw_obj['has_relocations'],
            raw_obj['has_resources'],
            raw_obj['has_signature'],
            raw_obj['has_tls'],
            raw_obj['symbols']
        ], dtype=np.float32)


class HeaderFileInfo(FeatureType):
    """ COFF header and optional header. """

    name = 'header'
    dim = (1 + 10 + 10) + (10 + 10 + 10 + 11)

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, raw_bytes, lief_binary):
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
        if lief_binary is None:
            return raw_obj

        coff_header = lief_binary.header
        raw_obj['coff']['timestamp'] = coff_header.time_date_stamps
        raw_obj['coff']['machine'] = str(coff_header.machine).split('.')[-1]
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1]
                                              for c in coff_header.characteristics_list]

        optional_header = lief_binary.optional_header
        raw_obj['optional']['subsystem'] = str(optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [str(c).split('.')[-1]
                                                      for c in optional_header.dll_characteristics_lists]
        raw_obj['optional']['magic'] = str(optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = optional_header.minor_linker_version
        raw_obj['optional']['major_operating_system_version'] = optional_header.major_operating_system_version
        raw_obj['optional']['minor_operating_system_version'] = optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            # coff header
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            # optional header
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)


class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()
        # 0x20 - 0x7f, at least 5 character
        self.all_strings_re = re.compile(b'[\x20-\x7f]{5,}')
        # 'C:\' or 'c:\'
        self.paths_re = re.compile(b'c:\\\\', re.IGNORECASE)
        # http:// or https://
        self.urls_re = re.compile(b'https?://', re.IGNORECASE)
        # HKEY_
        self.registries_re = re.compile(b'HKEY_')
        # MZ
        self._mz = re.compile(b'MZ')

    def raw_features(self, raw_bytes, lief_binary):
        all_strings = self.all_strings_re.findall(raw_bytes)
        if all_strings:
            string_lengths = [len(s) for s in all_strings]
            avg_length = sum(string_lengths) * 1.0 / len(string_lengths)
            shifted_string = [b - ord(b'\x20') for b in b''.join(all_strings)]
            c = np.bincount(shifted_string, minlength=96)
            total = c.sum()
            p = c.astype(np.float32) / total
            i = np.nonzero(c)[0]
            entropy = np.sum(-p[i] * np.log2(p[i]))
        else:
            avg_length = 0
            c = np.zeros((96,), dtype=np.float32)
            entropy = 0
            total = 0

        return {
            'numstrings': len(all_strings),
            'avlength': avg_length,
            'printabledist': c.tolist(),
            'printables': int(total),
            'entropy': float(entropy),
            'paths': len(self.paths_re.findall(raw_bytes)),
            'urls': len(self.urls_re.findall(raw_bytes)),
            'registry': len(self.registries_re.findall(raw_bytes)),
            'MZ': len(self._mz.findall(raw_bytes))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'],
            raw_obj['avlength'],
            raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor,
            raw_obj['entropy'],
            raw_obj['paths'],
            raw_obj['urls'],
            raw_obj['registry'],
            raw_obj['MZ']
        ]).astype(np.float32)


class FeatureExtractor(object):
    """ Extract features from a PE file. """

    features = [
        ByteHistogram(),
        ByteEntropyHistogram(),
        StringExtractor(),
        GeneralFileInfo(),
        HeaderFileInfo(),
        SectionInfo(),
        ImportsInfo(),
        ExportsInfo()
    ]
    dim = sum([fe.dim for fe in features])

    def raw_features(self, file_path):
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        try:
            lief_binary = lief.parse(file_path)
        except (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
            logging.error("lief error: {}".format(e))
            lief_binary = None

        features = {
            "sha256": hashlib.sha256(raw_bytes).hexdigest()
        }
        features.update({
            feature.name: feature.raw_features(raw_bytes, lief_binary)
            for feature in self.features
        })
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [
            features.process_raw_features(raw_obj[features.name])
            for features in self.features
        ]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, file_path):
        return self.process_raw_features(self.raw_features(file_path))
