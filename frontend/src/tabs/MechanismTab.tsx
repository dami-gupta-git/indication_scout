// Mechanism tab: summary, molecular targets, MoA table, and mechanism-derived
// repurposing candidates.

import type { SupervisorOutput } from "../types";

export function MechanismTab({ result }: { result: SupervisorOutput }) {
  const mech = result.mechanism;
  if (mech === null) {
    return <p className="muted">Mechanism analysis not run.</p>;
  }

  const targets = Object.keys(mech.drug_targets).sort();

  return (
    <div className="mechanism">
      <h3>Mechanistic analysis</h3>
      {mech.summary && <p>{mech.summary}</p>}

      <div className="mech-cols">
        <div className="mech-col targets">
          <h4>Molecular targets</h4>
          {targets.length > 0 ? (
            <ul>
              {targets.map((t) => (
                <li key={t}>
                  <code>{t}</code>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No targets identified.</p>
          )}
        </div>

        <div className="mech-col moa">
          <h4>Mechanisms of action</h4>
          {mech.mechanisms_of_action.length > 0 ? (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Mechanism</th>
                    <th>Action type</th>
                    <th>Targets</th>
                  </tr>
                </thead>
                <tbody>
                  {mech.mechanisms_of_action.map((moa, i) => (
                    <tr key={`${moa.mechanism_of_action}-${i}`}>
                      <td>{moa.mechanism_of_action}</td>
                      <td>{moa.action_type ?? ""}</td>
                      <td>
                        {moa.target_symbols.length > 0
                          ? moa.target_symbols.join(", ")
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">No mechanisms of action recorded.</p>
          )}
        </div>
      </div>

      {mech.candidates.length > 0 && (
        <>
          <h4>Repurposing candidates from mechanism</h4>
          <div className="mech-candidates">
            {mech.candidates.map((c, i) => (
              <div className="card" key={`${c.target_symbol}-${c.disease_name}-${i}`}>
                <strong>
                  {c.target_symbol} ({c.action_type}) → {c.disease_name}
                </strong>
                {c.disease_description && (
                  <p className="caption">{c.disease_description}</p>
                )}
                {c.target_function && (
                  <p className="caption">
                    <em>Target function:</em> {c.target_function}
                  </p>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
