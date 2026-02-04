"""Hard research-focused RAG evaluation."""
import requests, json, time, sys

BASE = 'http://localhost:8003'
TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJlbWFpbCI6Imlza2x2LmFnQGdtYWlsLmNvbSIsImV4cCI6MTc3MDIwNDgyOX0.262cORBX5dO-l9InK4x6-blsbn2hUv03M5JGUeXtwek'
HEADERS = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
PROJECT_ID = 57

r = requests.post(f'{BASE}/api/conversations', headers=HEADERS, json={'project_id': PROJECT_ID, 'title': 'Hard Eval Run'})
conv = r.json()
conv_id = conv['id']
print(f'Conversation ID: {conv_id}')

QA_PAIRS = [
    # === EEGPT Paper - Deep methodology questions ===
    {
        'id': 1,
        'question': 'What is the dual self-supervised pretraining method used in EEGPT? Describe both branches and how they differ from standard BERT-style masking.',
        'expected_keywords': ['spatio-temporal', 'alignment', 'mask', 'reconstruct', 'representation'],
        'source': 'EEGPT'
    },
    {
        'id': 2,
        'question': 'How many parameters does the EEGPT model have, and what masking ratios were used during pretraining for time and channel patches?',
        'expected_keywords': ['10', 'million', '50%', '80%'],
        'source': 'EEGPT'
    },
    {
        'id': 3,
        'question': 'What EEG datasets were used to pretrain EEGPT and what downstream benchmarks were used for evaluation? List specific dataset names.',
        'expected_keywords': ['PhysioMI', 'BCIC', 'TUAB', 'SEED', 'Sleep'],
        'source': 'EEGPT'
    },
    {
        'id': 4,
        'question': 'In the EEGPT ablation study, what happens to performance when the alignment loss (LA) is removed? Report the specific BCIC-2A balanced accuracy.',
        'expected_keywords': ['0.52', 'ablat', 'alignment'],
        'source': 'EEGPT'
    },
    {
        'id': 5,
        'question': 'What hardware setup was used to train EEGPT and what optimizer and learning rate schedule were employed?',
        'expected_keywords': ['3090', 'AdamW', 'OneCycle', '200 epoch'],
        'source': 'EEGPT'
    },

    # === Tree Reasoning Paper - Architecture and approach ===
    {
        'id': 6,
        'question': 'What is the core argument of the Tree Reasoning paper against standard RAG chunking for PDFs? What specific failure mode does it identify?',
        'expected_keywords': ['structure', 'semantic', 'similar', 'result', 'method', 'vector'],
        'source': 'Tree Reasoning'
    },
    {
        'id': 7,
        'question': 'How does the Tree Reasoning approach handle documents that lack detectable structure, such as scanned receipts?',
        'expected_keywords': ['fallback', 'standard chunking', 'vision', 'low-text'],
        'source': 'Tree Reasoning'
    },

    # === AI in Kazakhstan Report - Specific data points ===
    {
        'id': 8,
        'question': 'According to the Kazakhstan AI Country Report, what is the name and ranking of the supercomputer launched in Kazakhstan in 2025? How many GPUs does it use?',
        'expected_keywords': ['Alem', '86', 'TOP500', '512', 'H200'],
        'source': 'AI Kazakhstan'
    },
    {
        'id': 9,
        'question': 'What percentage of jobs in Kazakhstan can be fully or partially replaced by AI, and what percentage can be augmented according to the report?',
        'expected_keywords': ['54%', '41%', '5%'],
        'source': 'AI Kazakhstan'
    },
    {
        'id': 10,
        'question': 'What is Cerebra AI and what has it achieved in clinical deployment? Mention specific countries and number of pilot projects.',
        'expected_keywords': ['CT', 'pilot', '55', '3 countr', 'certif'],
        'source': 'AI Kazakhstan'
    },
    {
        'id': 11,
        'question': 'What is the Kazakhstan government target for IT exports, and what organization is leading the startup ecosystem expansion?',
        'expected_keywords': ['5', 'billion', 'Astana Hub', '2029'],
        'source': 'AI Kazakhstan'
    },

    # === Innovative Architectural Solutions (BULT platform paper) ===
    {
        'id': 12,
        'question': 'What container orchestration tool does the BULT platform use instead of Kubernetes, and what database technologies does it employ?',
        'expected_keywords': ['Nomad', 'HashiCorp', 'PostgreSQL', 'JuiceFS'],
        'source': 'BULT Platform'
    },
    {
        'id': 13,
        'question': 'What international security and compliance standards does the BULT platform address according to the paper?',
        'expected_keywords': ['GDPR', 'ISO 27001'],
        'source': 'BULT Platform'
    },

    # === Cross-document research synthesis ===
    {
        'id': 14,
        'question': 'Both EEGPT and the Tree Reasoning paper propose novel approaches to existing problems. Compare their core innovations: what fundamental limitation does each paper address in its respective field?',
        'expected_keywords': ['EEG', 'self-supervised', 'variabilit', 'structure', 'chunk', 'PDF'],
        'source': 'cross-document'
    },
    {
        'id': 15,
        'question': 'Based on the Kazakhstan AI report and the BULT platform paper, what role does cloud infrastructure play in Kazakhstan AI ecosystem development? What specific compute capabilities are mentioned?',
        'expected_keywords': ['cloud', 'GPU', 'microservice', 'supercomputer', 'infrastructure'],
        'source': 'cross-document'
    },
]

results = []
total = len(QA_PAIRS)

for qa in QA_PAIRS:
    qid = qa['id']
    question = qa['question']
    keywords = qa['expected_keywords']
    print(f'\n--- Q{qid}/{total}: {question[:80]}...')

    try:
        r = requests.post(
            f'{BASE}/api/chat',
            headers={**HEADERS, 'Accept': 'text/event-stream'},
            json={'project_id': PROJECT_ID, 'conversation_id': conv_id, 'message': question},
            stream=True,
            timeout=120
        )

        answer = ''
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    answer += chunk.get('content', '')
                except json.JSONDecodeError:
                    pass

        answer_lower = answer.lower()
        matched = [kw for kw in keywords if kw.lower() in answer_lower]
        score = len(matched) / len(keywords) if keywords else 0
        passed = score >= 0.5

        results.append({
            'id': qid,
            'question': question[:90],
            'score': score,
            'passed': passed,
            'matched': matched,
            'missed': [kw for kw in keywords if kw.lower() not in answer_lower],
            'answer_length': len(answer),
            'source': qa['source']
        })

        status = 'PASS' if passed else 'FAIL'
        print(f'  {status} | Score: {score:.0%} | Matched: {matched}')
        if [kw for kw in keywords if kw.lower() not in answer_lower]:
            print(f'  Missed: {[kw for kw in keywords if kw.lower() not in answer_lower]}')
        print(f'  Answer length: {len(answer)} chars')

    except Exception as e:
        print(f'  ERROR: {e}')
        results.append({
            'id': qid, 'question': question[:90], 'score': 0, 'passed': False,
            'matched': [], 'missed': keywords, 'answer_length': 0, 'source': qa['source']
        })

    time.sleep(1)

# Summary
print('\n' + '=' * 70)
print('HARD EVALUATION SUMMARY')
print('=' * 70)
passed_count = sum(1 for r in results if r['passed'])
total_score = sum(r['score'] for r in results) / len(results) * 100
print(f'Passed: {passed_count}/{total} ({passed_count/total*100:.1f}%)')
print(f'Average score: {total_score:.1f}%')
print()

# Group by source
sources = {}
for r in results:
    src = r['source']
    if src not in sources:
        sources[src] = []
    sources[src].append(r)

for src, items in sources.items():
    avg = sum(i['score'] for i in items) / len(items) * 100
    print(f'  [{src}] avg={avg:.0f}%')
    for r in items:
        status = 'PASS' if r['passed'] else 'FAIL'
        missed_str = ''
        if r.get('missed'):
            missed_str = f'\n         Missed: {r["missed"]}'
        print(f'    Q{r["id"]:2d} [{status}] {r["score"]:5.0%} | {r["question"][:65]}{missed_str}')
