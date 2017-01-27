import math

# static data
good_cands_ls = [(3, 23, 0), (11, 51, 0), (12, 73, 1), (20, 32, 0), (20, 50, 0), (20, 73, 1), (38, 32, 0), (38, 50, 0),
                 (38, 73, 1), (39, 73, 2), (47, 41, 0), (50, 22, 0), (55, 41, 0), (56, 31, 4), (56, 49, 4), (56, 50, 0),
                 (56, 73, 1), (57, 22, 1), (57, 73, 2), (82, 31, 4)]  # LIGHT SPLITTERS (20)
# good_cands_ls = [(3, 23, 0), (11, 51, 0), (12, 73, 1), (20, 32, 0), (49, 72, 4), (20, 73, 1), (38, 32, 0), (38, 50, 0),
#    bad!!!              (38, 73, 1), (39, 73, 2), (47, 41, 0), (50, 22, 0), (55, 41, 0), (56, 31, 4), (56, 49, 4), (56, 50, 0),
#                  (56, 73, 1), (57, 22, 1), (57, 73, 2), (82, 31, 4)]  # LIGHT SPLITTERS (20)
good_cands_os = [(20, 50, 0), (37, 22, 4), (37, 41, 0), (38, 22, 0), (38, 31, 4), (38, 50, 0), (55, 73, 0),
                 (56, 49, 4)]  # OXIDE SHIELDS (8)
num_cands = 18928


atomic_index =[3, 4, 5, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40,
                41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
anion_index = list(range(7))
cation_names= ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
anion_names = ['O3', 'O2N', 'ON2', 'N3', 'O2F', 'OFN', 'O2S']



# Utility classes

class Evaluator (object):
    def simple(self, gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
        stab_score = 0
        gap_dir_score = 0
        gap_ind_score = 0

        if (gap_dir >= 1.5 and gap_dir <= 3):
            gap_dir_score += 10

        if (gap_ind >= 1.5 and gap_ind <= 3):
            gap_ind_score += 10

        if heat_of_formation <= 0.5:
            stab_score += 5

        if heat_of_formation <= 0.2:
            stab_score += 5

        if (vb_dir >= 5.73):
            gap_dir_score += 5

        if (cb_dir <= 4.5):
            gap_dir_score += 5

        if (vb_ind >= 5.73):
            gap_ind_score += 5

        if (cb_ind <= 4.5):
            gap_ind_score += 5

        return max(gap_ind_score, gap_dir_score) + stab_score

    def complex(self, gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
        stab_score = 0
        gap_dir_score = 0
        gap_ind_score = 0

        if (gap_dir >= 1.5 and gap_dir <= 3):
            gap_dir_score += 10
        elif gap_dir == 0:
            gap_dir_score += 0
        else:
            gap_dir_score += 33 * self.gaussian_pdf(gap_dir, 2.25)

        if (gap_ind >= 1.5 and gap_ind <= 3):
            gap_ind_score += 10
        elif gap_ind == 0:
            gap_ind_score += 0
        else:
            gap_ind_score += 33 * self.gaussian_pdf(gap_ind, 2.25)

        if heat_of_formation <= 0.2:
            stab_score = 10
        else:
            stab_score = 20 * (1 - 1 / (1 + math.exp(((-heat_of_formation) + 0.2) * 3.5)))

        if vb_dir >= 5.73:
            gap_dir_score += 5
        else:
            distance = (5.73 - vb_dir) * 5
            gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if vb_ind >= 5.73:
            gap_ind_score += 5
        else:
            distance = (5.73 - vb_ind) * 5
            gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if cb_dir <= 4.5:
            gap_dir_score += 5
        else:
            distance = (cb_dir - 4.5) * 5
            gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if cb_ind <= 4.5:
            gap_ind_score += 5
        else:
            distance = (cb_ind - 4.5) * 5
            gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        return max(gap_ind_score, gap_dir_score) + stab_score

    def complex_product(self, gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
        stab_score = 0
        gap_dir_score = 0
        gap_ind_score = 0

        if (gap_dir >= 1.5 and gap_dir <= 3):
            gap_dir_score += 10
        elif gap_dir == 0:
            gap_dir_score += 0
        else:
            gap_dir_score += 33 * self.gaussian_pdf(gap_dir, 2.25)

        if (gap_ind >= 1.5 and gap_ind <= 3):
            gap_ind_score += 10
        elif gap_ind == 0:
            gap_ind_score += 0
        else:
            gap_ind_score += 33 * self.gaussian_pdf(gap_ind, 2.25)

        if heat_of_formation <= 0.2:
            stab_score = 10
        else:
            stab_score = 20 * (1 - 1 / (1 + math.exp(((-heat_of_formation) + 0.2) * 3.5)))

        if vb_dir >= 5.73:
            gap_dir_score += 5
        else:
            distance = (5.73 - vb_dir) * 5
            gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if vb_ind >= 5.73:
            gap_ind_score += 5
        else:
            distance = (5.73 - vb_ind) * 5
            gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if cb_dir <= 4.5:
            gap_dir_score += 5
        else:
            distance = (cb_dir - 4.5) * 5
            gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        if cb_ind <= 4.5:
            gap_ind_score += 5
        else:
            distance = (cb_ind - 4.5) * 5
            gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

        return max(gap_ind_score, gap_dir_score) * stab_score * 0.15

    def gaussian_pdf(self, x, mean=0, width=0.5):
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-width * (x - mean) * (x - mean))

class Converter (object):
    def __init__(self):
        self.atomic2name = dict(zip(atomic_index, cation_names))
        self.anion_index2name = dict(zip(anion_index, anion_names))

    def atomic_to_name(self, atomic_tuple):
        anion = atomic_tuple[2]
        A = atomic_tuple[1]
        B = atomic_tuple[0]

        return (self.atomic2name[A], self.atomic2name[B], self.anion_index2name[anion])

