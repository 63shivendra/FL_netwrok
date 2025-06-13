import os
import json
import csv

# === CONFIGURE THESE PATHS ===
data_dir   = "fhir"                   # root folder of your fhir/00/000/... hierarchy
output_csv = "simplified_ehr_data.csv"  # where to save the CSV

# -----------------------------------------------------------------------------
# 1. Discover all JSON files under data_dir
# -----------------------------------------------------------------------------
json_files = []
for root, _, files in os.walk(data_dir):
    for fname in files:
        if fname.lower().endswith(".json"):
            json_files.append(os.path.join(root, fname))
json_files.sort()
total_files = len(json_files)
print(f"Found {total_files} JSON patient files under {data_dir}")

# -----------------------------------------------------------------------------
# 2. Open CSV and write header
# -----------------------------------------------------------------------------
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "patient_id",
        "name",
        "gender",
        "birthDate",
        "race",
        "ethnicity",
        "maritalStatus",
        "language",
        "conditions",         # all diagnoses joined by '; '
        "primary_disease",    # first diagnosis
        "medications",        # all meds joined by '; '
        "encounters",         # count
        "observation_count",  # count
        "procedure_count"     # count
    ])

    # -----------------------------------------------------------------------------
    # 3. Process files in chunks
    # -----------------------------------------------------------------------------
    chunk_size = 500
    for start in range(0, total_files, chunk_size):
        end = min(start + chunk_size, total_files)
        rows = []

        for fp in json_files[start:end]:
            try:
                bundle = json.load(open(fp, 'r', encoding='utf-8'))
            except Exception:
                # skip files that can’t be read or parsed
                continue

            # ---- initialize per-patient variables ----
            patient_id      = ""
            name            = ""
            gender          = ""
            birthDate       = ""
            race            = ""
            ethnicity       = ""
            maritalStatus   = ""
            language        = ""
            conditions      = []
            medications     = []
            encounters      = 0
            obs_count       = 0
            proc_count      = 0

            # ---- iterate through Bundle entries ----
            for entry in bundle.get("entry", []):
                res = entry.get("resource", {})
                rt  = res.get("resourceType", "")

                # --- Patient demographics & extensions ---
                if rt == "Patient":
                    patient_id = res.get("id", "")
                    # name: combine given + family
                    if res.get("name"):
                        nm    = res["name"][0]
                        given = nm.get("given", [])
                        fam   = nm.get("family", "")
                        name  = (" ".join(given) + (" " + fam if fam else "")).strip()
                    gender    = res.get("gender", "")
                    birthDate = res.get("birthDate", "")

                    # look for US Core extensions (race, ethnicity)
                    for ext in res.get("extension", []):
                        url = ext.get("url", "")
                        vcc = ext.get("valueCodeableConcept", {})
                        txt = vcc.get("text", "")
                        if url.endswith("us-core-race"):
                            race = txt
                        elif url.endswith("us-core-ethnicity"):
                            ethnicity = txt

                    # maritalStatus
                    maritalStatus = res.get("maritalStatus", {}).get("text", "")

                    # first communication.language
                    comms = res.get("communication", [])
                    if comms:
                        language = comms[0].get("language", {}).get("text", "")

                # --- Conditions (diagnoses) ---
                elif rt == "Condition":
                    code = res.get("code", {})
                    txt  = code.get("text")
                    if not txt and code.get("coding"):
                        cd  = code["coding"][0]
                        txt = cd.get("display") or cd.get("code")
                    if txt:
                        conditions.append(txt)

                # --- Medications ---
                elif rt in ("MedicationRequest", "MedicationStatement", "MedicationOrder"):
                    mc  = res.get("medicationCodeableConcept", {})
                    txt = mc.get("text")
                    if not txt and mc.get("coding"):
                        md  = mc["coding"][0]
                        txt = md.get("display") or md.get("code")
                    if txt:
                        medications.append(txt)

                # --- Encounter count ---
                elif rt == "Encounter":
                    encounters += 1

                # --- Observation count ---
                elif rt == "Observation":
                    obs_count += 1

                # --- Procedure count ---
                elif rt == "Procedure":
                    proc_count += 1

            # ---- prepare CSV row ----
            cond_str  = "; ".join(conditions)
            primary   = conditions[0] if conditions else ""
            med_str   = "; ".join(medications)

            rows.append([
                patient_id,
                name,
                gender,
                birthDate,
                race,
                ethnicity,
                maritalStatus,
                language,
                cond_str,
                primary,
                med_str,
                encounters,
                obs_count,
                proc_count
            ])

        # write this chunk to CSV
        writer.writerows(rows)
        print(f"  • Processed {start+1}–{end} of {total_files}")

print(f"\nAll done! CSV saved to: {output_csv}")
