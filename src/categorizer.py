import re
import nltk
from textblob import TextBlob
import numpy as np
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag


class DocumentCategorizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.document_type_patterns = {
            'Email': [
                r'\b(dear|hello|hi|regards|sincerely|best wishes|from:|to:|subject:|cc:|bcc:)\b',
                r'\b(email|message|correspondence|reply|forward)\b',
                r'@\w+\.\w+'
            ],
            'Report': [
                r'\b(executive summary|introduction|methodology|findings|conclusions|recommendations)\b',
                r'\b(analysis|research|study|investigation|evaluation)\b',
                r'\b(table of contents|appendix|bibliography|references)\b'
            ],
            'Proposal': [
                r'\b(proposal|bid|quotation|estimate|budget|timeline)\b',
                r'\b(scope of work|deliverables|objectives|goals)\b',
                r'\b(cost|pricing|payment|terms and conditions)\b'
            ],
            'Meeting Notes': [
                r'\b(meeting|agenda|minutes|action items|attendees)\b',
                r'\b(discussed|decided|agreed|next steps|follow up)\b',
                r'\b(date:|time:|location:|participants:)\b'
            ],
            'Contract': [
                r'\b(agreement|contract|terms|conditions|parties)\b',
                r'\b(whereas|therefore|shall|hereby|notwithstanding)\b',
                r'\b(signature|executed|effective date|termination)\b'
            ],
            'Manual': [
                r'\b(instructions|procedure|step|guide|manual|tutorial)\b',
                r'\b(how to|getting started|configuration|setup)\b',
                r'\b(warning|caution|note|important|tip)\b'
            ],
            'Policy': [
                r'\b(policy|procedure|guidelines|standards|compliance)\b',
                r'\b(must|shall|required|mandatory|prohibited)\b',
                r'\b(violation|penalty|enforcement|review)\b'
            ]
        }
        
        self.business_categories = {
            'Finance': [
                'budget', 'revenue', 'profit', 'loss', 'investment', 'financial', 'accounting',
                'expense', 'cost', 'income', 'cash flow', 'balance sheet', 'roi', 'tax'
            ],
            'HR': [
                'employee', 'staff', 'recruitment', 'hiring', 'performance', 'training',
                'benefits', 'salary', 'payroll', 'personnel', 'human resources', 'team'
            ],
            'Marketing': [
                'campaign', 'brand', 'advertising', 'promotion', 'customer', 'market',
                'sales', 'lead', 'conversion', 'social media', 'seo', 'content marketing'
            ],
            'Operations': [
                'process', 'workflow', 'efficiency', 'productivity', 'quality', 'supply chain',
                'logistics', 'manufacturing', 'delivery', 'inventory', 'operations'
            ],
            'IT': [
                'software', 'hardware', 'system', 'network', 'security', 'database',
                'server', 'cloud', 'application', 'technology', 'infrastructure', 'support'
            ],
            'Legal': [
                'contract', 'agreement', 'compliance', 'regulation', 'law', 'legal',
                'litigation', 'intellectual property', 'patent', 'trademark', 'copyright'
            ],
            'Strategy': [
                'strategic', 'planning', 'vision', 'mission', 'objectives', 'goals',
                'roadmap', 'competitive', 'market analysis', 'swot', 'kpi', 'metrics'
            ]
        }
        
    def analyze_document(self, content, filename):
        """Comprehensive document analysis for categorization and tagging"""
        analysis = {
            'filename': filename,
            'content': content,
            'word_count': len(content.split()),
            'sentence_count': len(sent_tokenize(content)),
            'readability': self._calculate_readability(content),
            'sentiment': self._analyze_sentiment(content),
            'document_type': self._classify_document_type(content),
            'business_category': self._classify_business_category(content),
            'key_topics': self._extract_key_topics(content),
            'tags': self._generate_tags(content),
            'entities': self._extract_entities(content),
            'urgency_level': self._assess_urgency(content),
            'complexity_score': self._calculate_complexity(content)
        }
        
        return analysis
    
    def _calculate_readability(self, text):
        """Calculate Flesch Reading Ease score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        syllables = sum([self._count_syllables(word) for word in words])
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
        return round(max(0, min(100, score)), 1)
    
    def _count_syllables(self, word):
        """Simple syllable counting"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_sentiment(self, text):
        """Analyze document sentiment"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment_label = 'Positive'
        elif polarity < -0.1:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'label': sentiment_label,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3)
        }
    
    def _classify_document_type(self, content):
        """Classify document type based on patterns"""
        content_lower = content.lower()
        type_scores = {}
        
        for doc_type, patterns in self.document_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            type_scores[doc_type] = score
        
        if max(type_scores.values()) == 0:
            return 'General Document'
        
        return max(type_scores, key=type_scores.get)
    
    def _classify_business_category(self, content):
        """Classify business category based on keywords"""
        content_lower = content.lower()
        category_scores = {}
        
        for category, keywords in self.business_categories.items():
            score = 0
            for keyword in keywords:
                score += content_lower.count(keyword.lower())
            category_scores[category] = score
        
        if max(category_scores.values()) == 0:
            return 'General'
        
        return max(category_scores, key=category_scores.get)
    
    def _extract_key_topics(self, content, num_topics=3):
        """Extract key topics using keyword frequency"""
        words = word_tokenize(content.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 3]
        
        if len(words) < 10:
            return []
        
        # Get most frequent meaningful words
        word_freq = Counter(words)
        common_words = word_freq.most_common(num_topics * 2)
        
        # Filter out very common words
        topics = []
        for word, freq in common_words:
            if freq >= 2 and len(topics) < num_topics:
                topics.append({
                    'topic': word.title(),
                    'frequency': freq,
                    'relevance': round(freq / len(words) * 100, 2)
                })
        
        return topics
    
    def _generate_tags(self, content):
        """Generate relevant tags for the document"""
        tags = set()
        content_lower = content.lower()
        
        # Extract important nouns and phrases
        words = word_tokenize(content)
        pos_tags = pos_tag(words)
        
        # Get important nouns
        nouns = [word.lower() for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(word) > 3]
        noun_freq = Counter(nouns)
        
        for noun, freq in noun_freq.most_common(10):
            if freq >= 2 and noun not in self.stop_words:
                tags.add(noun.title())
        
        # Add category-based tags
        for category, keywords in self.business_categories.items():
            for keyword in keywords[:5]:  # Check top 5 keywords per category
                if keyword.lower() in content_lower:
                    tags.add(keyword.title())
        
        # Add document type as tag
        doc_type = self._classify_document_type(content)
        if doc_type != 'General Document':
            tags.add(doc_type)
        
        return list(tags)[:15]  # Limit to 15 tags
    
    def _extract_entities(self, content):
        """Extract named entities (simple approach)"""
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content),
            'phone_numbers': re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}\b', content),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b', content),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content),
            'money': re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', content)
        }
        
        # Filter out empty lists
        return {k: v for k, v in entities.items() if v}
    
    def _assess_urgency(self, content):
        """Assess document urgency based on keywords"""
        urgency_keywords = {
            'high': ['urgent', 'asap', 'immediate', 'critical', 'emergency', 'deadline', 'rush'],
            'medium': ['soon', 'priority', 'important', 'needed', 'request'],
            'low': ['when possible', 'eventually', 'future', 'consider', 'maybe']
        }
        
        content_lower = content.lower()
        scores = {'high': 0, 'medium': 0, 'low': 0}
        
        for level, keywords in urgency_keywords.items():
            for keyword in keywords:
                scores[level] += content_lower.count(keyword)
        
        if scores['high'] > 0:
            return 'High'
        elif scores['medium'] > 0:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_complexity(self, content):
        """Calculate document complexity score"""
        sentences = sent_tokenize(content)
        words = word_tokenize(content)
        
        if not sentences or not words:
            return 0
        
        # Factors affecting complexity
        avg_sentence_length = len(words) / len(sentences)
        long_words = len([word for word in words if len(word) > 6])
        long_word_ratio = long_words / len(words) if words else 0
        
        # Complexity score (0-10)
        complexity = min(10, (avg_sentence_length / 5) + (long_word_ratio * 10))
        return round(complexity, 1)

    def categorize_documents(self, documents):
        """Categorize a list of documents"""
        categorized_docs = []
        
        for doc in documents:
            analysis = self.analyze_document(doc.get('content', ''), doc.get('filename', ''))
            
            # Merge original document data with analysis
            categorized_doc = {**doc, **analysis}
            categorized_docs.append(categorized_doc)
        
        return categorized_docs

    def get_category_summary(self, documents):
        """Get summary statistics for categories and tags"""
        if not documents:
            return {}
        
        categories = [doc.get('business_category', 'General') for doc in documents]
        doc_types = [doc.get('document_type', 'Unknown') for doc in documents]
        all_tags = []
        for doc in documents:
            all_tags.extend(doc.get('tags', []))
        
        return {
            'category_distribution': dict(Counter(categories)),
            'document_type_distribution': dict(Counter(doc_types)),
            'top_tags': dict(Counter(all_tags).most_common(20)),
            'total_documents': len(documents),
            'avg_readability': round(np.mean([doc.get('readability', 0) for doc in documents]), 1),
            'sentiment_distribution': dict(Counter([doc.get('sentiment', {}).get('label', 'Unknown') for doc in documents]))
        }