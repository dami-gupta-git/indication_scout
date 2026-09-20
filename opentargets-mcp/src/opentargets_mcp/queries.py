"""GraphQL documents. Every field here was confirmed against the live API on 2026-09-19."""

SEARCH_QUERY = """
query($q: String!, $entities: [String!]!, $size: Int!) {
    search(queryString: $q, entityNames: $entities, page: {index: 0, size: $size}) {
        hits { id entity name description }
    }
}
"""

TARGET_PROFILE_QUERY = """
query($id: String!) {
    target(ensemblId: $id) {
        id approvedSymbol approvedName biotype functionDescriptions
        tractability { label modality value }
        geneticConstraint { constraintType score oe oeLower oeUpper }
        safetyLiabilities {
            event datasource literature
            effects { direction dosing }
        }
    }
}
"""

TARGET_DISEASES_QUERY = """
query($id: String!, $size: Int!) {
    target(ensemblId: $id) {
        id approvedSymbol
        associatedDiseases(page: {index: 0, size: $size}) {
            count
            rows {
                disease { id name therapeuticAreas { id name } }
                score
                datatypeScores { id score }
            }
        }
    }
}
"""

DISEASE_TARGETS_QUERY = """
query($id: String!, $size: Int!) {
    disease(efoId: $id) {
        id name
        associatedTargets(page: {index: 0, size: $size}) {
            count
            rows {
                target { id approvedSymbol approvedName }
                score
                datatypeScores { id score }
            }
        }
    }
}
"""

TARGET_DRUGS_QUERY = """
query($id: String!) {
    target(ensemblId: $id) {
        id approvedSymbol
        drugAndClinicalCandidates {
            count
            rows {
                maxClinicalStage
                drug {
                    id name drugType
                    mechanismsOfAction { rows { mechanismOfAction actionType } }
                }
                diseases { disease { id name } }
            }
        }
    }
}
"""

DISEASE_DRUGS_QUERY = """
query($id: String!) {
    disease(efoId: $id) {
        id name
        drugAndClinicalCandidates {
            count
            rows {
                maxClinicalStage
                drug {
                    id name drugType
                    mechanismsOfAction { rows { mechanismOfAction actionType } }
                }
            }
        }
    }
}
"""

EVIDENCE_QUERY = """
query($id: String!, $efoIds: [String!]!, $size: Int!) {
    target(ensemblId: $id) {
        id approvedSymbol
        evidences(efoIds: $efoIds, size: $size) {
            count
            rows {
                datatypeId datasourceId score
                directionOnTarget directionOnTrait
                disease { id name }
                literature
            }
        }
    }
}
"""
