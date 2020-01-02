import copy
import numpy as np


class FaceBox:
    def __init__(self, box, landmarks):
        self.box = box
        self.landmarks = landmarks

        self.center = np.array([
            (self.box[3] + self.box[1]) / 2,
            (self.box[2] + self.box[0]) / 2], dtype=np.int32)

    def area(self):
        return (self.box[3] - self.box[1]) * (self.box[2] - self.box[0])

    def center_offset(self, other):
        return np.sqrt(np.sum(np.square(self.center - other.center)))

    def landmark_offset(self, other):
        return np.mean(np.abs(self.landmarks - other.landmarks))

    def dimension_offset(self, other):
        return np.sum(np.abs(self.box - other.box))

    def box_area_diff(self, other):
        return np.abs(self.area() - other.area())


class Person:
    def __init__(self, id, n_cached=30):
        self.id = id
        self.n_cached = n_cached
        self.ind = 0
        self.face_boxs = []

    def add_face_box(self, face_box):
        self.face_boxs = [face_box] + self.face_boxs
        if len(self.face_boxs) > self.n_cached:
            self.face_boxs = self.face_boxs[:self.n_cached]

    def get_face_boxs(self):
        return self.face_boxs

    def set_face_boxs(self, face_boxs):
        self.face_boxs = face_boxs


class SimpleFaceTracking:
    def __init__(self, n_cached=30, moving_threshold=5):
        self.n_cached = n_cached
        self.moving_threshold = moving_threshold
        self.persons = {}
        self.count_unk = 0

    def update(self, boxs, landmarks, identities):
        unk_boxs = []
        unk_box_candidates = []
        detected_id = []
        refined_identities = []

        for box, landmark, identity in zip(boxs, landmarks, identities):
            face_box = FaceBox(box[:4], landmark)
            if identity in self.persons:
                # Update identity's face box
                self.persons[identity].add_face_box(face_box)
                detected_id.append(identity)
            else:
                # Get identity's candidates for unk face box
                unk_boxs.append([identity, face_box])
                unk_box_candidates.append(self.get_candidates(face_box))

        for i in range(len(unk_boxs)):
            candidate_found = False
            unk_box_id, face_box = unk_boxs[i]
            # Find the most probably identity for the given unk face box
            for candidate_id, diff in unk_box_candidates[i]:
                if candidate_id in detected_id or \
                        candidate_id not in self.persons or \
                        (unk_box_id is not None and
                         not candidate_id.startswith('UNK')):
                    continue

                # Find candidate conflict with another unk face boxs
                flag = True
                for j in range(i + 1, len(unk_boxs)):
                    for cid, d in unk_box_candidates[j]:
                        if candidate_id != cid:
                            continue
                        if diff > d:
                            flag = False
                            break
                    if not flag:
                        break

                # No conflict, update candidate with this unk face box
                if flag:
                    candidate_found = True
                    # Face (box) id return by face recognition module is None
                    if unk_box_id is None:
                        detected_id.append(candidate_id)
                        self.persons[candidate_id].add_face_box(face_box)
                        refined_identities.append(candidate_id)
                    else:
                        detected_id.append(unk_box_id)
                        new_person = Person(unk_box_id, self.n_cached)
                        new_person.set_face_boxs(copy.deepcopy(
                            self.persons[candidate_id].get_face_boxs()))
                        new_person.add_face_box(face_box)
                        self.persons[unk_box_id] = new_person
                        del self.persons[candidate_id]
                    break

            # No suitable candidate found, create new identity
            if not candidate_found:
                if unk_box_id is None:
                    self.count_unk += 1
                    person_id = f'UNK_{self.count_unk}'
                    refined_identities.append(person_id)
                else:
                    person_id = unk_box_id

                detected_id.append(person_id)
                new_person = Person(person_id, self.n_cached)
                new_person.add_face_box(face_box)
                self.persons[person_id] = new_person

        # Update the rest of identities with None box
        del_identities = []
        for person_id, person in self.persons.items():
            if person_id in detected_id:
                continue
            person.add_face_box(None)

            # Delete identity if not in the screen for n_cached frames
            if len(person.get_face_boxs()) != self.n_cached:
                continue
            all_none = True
            for box in person.get_face_boxs():
                if box is not None:
                    all_none = False
                    break
            if all_none:
                del_identities.append(person_id)

        for pid in del_identities:
            del self.persons[pid]

        ind = 0
        for i in range(len(identities)):
            if identities[i] is None:
                identities[i] = refined_identities[ind]
                ind += 1

        assert ind == len(refined_identities)
        return identities

    def get_candidates(self, face_box):
        candidates = []
        for pid, person in self.persons.items():
            min_diff = 1000000
            for i, box in enumerate(person.get_face_boxs()):
                if box is None:
                    continue
                offset = face_box.center_offset(box)
                if offset < self.moving_threshold:
                    area_diff = face_box.box_area_diff(box)
                    landmark_diff = face_box.landmark_offset(box)
                    overall_diff = 1.4 * offset + 2.2 * area_diff + 0.1 * i
                    if overall_diff < min_diff:
                        min_diff = overall_diff

            if min_diff != 1000000:
                candidates.append([pid, min_diff])
        return sorted(candidates, key=lambda x: x[1])
