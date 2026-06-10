import random
class prompt():
    def __init__(self, text, acc):
        self.text = text
        self.acc = acc


class island():
    def __init__(self):
        self.texts = []
        self.accs = []
    def add_prompt(self, prompt):
        self.texts.append(prompt.text)
        self.accs.append(prompt.acc)
    def sort_prompts(self):
        combined = list(zip(self.texts, self.accs))
        combined.sort(key=lambda x: x[1], reverse=True)
        text_list, acc_list = zip(*combined)
        return text_list, acc_list
    def get_prompts(self):
        rand_idx = random.sample(range(len(self.texts)), 2)
        # if self.accs[rand_idx[0]] > self.accs[rand_idx[1]]:
        #     rand_idx[0], rand_idx[1] = rand_idx[1], rand_idx[0]
        return self.texts[rand_idx[0]] + ' \n' + self.texts[rand_idx[1]] + ' \n'
    def get_prompts_fun(self):
        combined = list(zip(self.texts, self.accs))
        # Sort by accs
        combined.sort(key=lambda x: x[1], reverse=True)
        # Return texts corresponding to the two highest accs values
        return combined[1][0] + '\n' + combined[0][0] + '\n'
    def transfer(self):
        text_list, acc_list = self.sort_prompts()
        self.texts = [text_list[0], text_list[1]]
        self.accs = [acc_list[0], acc_list[1]]
    def is_repeat(self, text):
        return text in self.texts
    def update(self,new_islands):
        self.texts = new_islands.texts
        self.accs = new_islands.accs
    def remove(self):
        if len(self.texts) > 6:
            combined = list(zip(self.accs, self.texts))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            top_6 = sorted_combined[-6:]
            self.accs, self.texts = map(list, zip(*top_6))
    def remove_lmx(self):
        while len(self.texts) > 6:
            rand_idx = random.sample(range(len(self.texts)), 2)
            if self.accs[rand_idx[0]] > self.accs[rand_idx[1]]:
                del_idx = rand_idx[1]
            else:
                del_idx = rand_idx[0]
            del self.texts[del_idx]
            del self.accs[del_idx]

class island_group():
    def __init__(self):
        self.islands = []
    def add_island(self, island):
        self.islands.append(island)
    def sort_transfer_islands(self):
        # Sort islands list based on the maximum value of island.accs
        self.islands.sort(key=lambda island: max(island.accs), reverse=True)

        # Calculate the split point for "good" and "bad"
        split_point = len(self.islands) // 2

        # Split islands list into "good" and "bad" parts
        good_islands = self.islands[:split_point]
        bad_islands = self.islands[split_point:]

        # Return indices of "good" and "bad" islands
        good_indices = [self.islands.index(island) for island in good_islands]
        bad_indices = [self.islands.index(island) for island in bad_islands]
        for bad_index in bad_indices:
            # Clear text and accs of bad_island
            self.islands[bad_index].text = []
            self.islands[bad_index].accs = []

            # Randomly select an island from good_indices
            good_index = random.choice(good_indices)

            # Copy the first two individuals from good_island
            tex, acc = self.islands[good_index].transfer()
            self.islands[bad_index].text.extend(tex)
            self.islands[bad_index].accs.extend(acc)
    def island_update(self):
        acc_list = []
        for island in self.islands:
            acc_list.append(max(island.accs))
        median = sorted(acc_list)[len(acc_list) // 2]
        lower_half_indices = [i for i, acc in enumerate(acc_list) if acc < median]
        upper_half_indices = [i for i, acc in enumerate(acc_list) if acc >= median]
        for i, j in zip(lower_half_indices, upper_half_indices):
            self.islands[i].update(self.islands[j])
     