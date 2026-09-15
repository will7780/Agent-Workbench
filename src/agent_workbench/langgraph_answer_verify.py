"""Bounded diagnostics, not a business acceptance score or semantic fact check."""


def verify_final_answer(*, final_response, tool_calls, observations,
                        failed_steps, completed_steps, knowledge_search=None):
    from .knowledge_citation import verify_knowledge_citations
    citation = verify_knowledge_citations(final_response, knowledge_search)
    issues = list(citation.get('issues') or [])
    if not str(final_response or '').strip():
        issues.append({'type': 'final_response_missing'})
    return {'passed': not issues, 'issues': issues,
            'semantic_verification': 'not_verified',
            'evidence_step_ids': [o['step_id'] for o in observations if o.get('step_id')],
            'citation_verification': citation}
