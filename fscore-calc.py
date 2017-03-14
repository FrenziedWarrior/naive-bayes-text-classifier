TEST_LABELS = 'test-labels-set1.txt'
OUTPUT_FILE = 'nboutput.txt'


classes = {
    'd':
    {'tp':0, 'fp':0, 'fn':0}, 
    
    't':
    {'tp':0, 'fp':0, 'fn':0},

    'p':
    {'tp':0, 'fp':0, 'fn':0}, 
    
    'n':
    {'tp':0, 'fp':0, 'fn':0}

}


print('Precision Recall F1')

with open(TEST_LABELS, 'r') as labels, open(OUTPUT_FILE, 'r') as output:
    for op_line, ref_line in zip(output, labels):
        op_line = op_line.strip('\n\r').split(' ')
        
        if op_line[1] in ref_line:
            if op_line[1] == 'deceptive':
                classes['d']['tp'] += 1
            elif op_line[1] == 'truthful':
                classes['t']['tp'] += 1
        elif 'deceptive' == op_line[1]:
            classes['d']['fp'] += 1
            classes['t']['fn'] += 1
        elif 'truthful' == op_line[1]:
            classes['d']['fn'] += 1
            classes['t']['fp'] += 1
        
        if op_line[2] in ref_line:
            if op_line[2] == 'positive':
                classes['p']['tp'] += 1
            elif op_line[2] == 'negative':
                classes['n']['tp'] += 1
        elif 'positive' == op_line[2]:
            classes['p']['fp'] += 1
            classes['n']['fn'] += 1
        elif 'negative' == op_line[2]:
            classes['n']['fp'] += 1
            classes['p']['fn'] += 1
        
precision = {}
recall = {}
fscore = {}

print(classes)

score_sum = 0
for c in classes.keys():
    precision[c] = classes[c]['tp']/(classes[c]['tp'] + classes[c]['fp'])
    recall[c] = classes[c]['tp']/(classes[c]['tp'] + classes[c]['fn'])
    fscore[c] = (2*precision[c]*recall[c]) / (precision[c]+recall[c])
    score_sum += fscore[c]/4
    print(c, precision[c], recall[c], fscore[c])

print('Weighted Avg: %f' % score_sum)