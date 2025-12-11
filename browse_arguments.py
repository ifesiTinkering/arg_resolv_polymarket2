#!/usr/bin/env python3
"""
Browse Arguments - Interactive Chat UI
View arguments as chat conversations with clickable fact-checking
"""

import gradio as gr
from storage import ArgumentStorage
from datetime import datetime
import json

storage = ArgumentStorage()

# Emotion labels (no emojis)
EMOTION_LABELS = {
    'calm': 'CALM',
    'confident': 'CONFIDENT',
    'defensive': 'DEFENSIVE',
    'dismissive': 'DISMISSIVE',
    'passionate': 'PASSIONATE',
    'frustrated': 'FRUSTRATED',
    'angry': 'ANGRY',
    'sarcastic': 'SARCASTIC',
    'unknown': 'UNKNOWN'
}

def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable string"""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return iso_timestamp

def get_arguments_list():
    """Get formatted list of all arguments"""
    arguments = storage.list_arguments(limit=100)

    if not arguments:
        return "No arguments recorded yet. Start by recording from the Raspberry Pi!"

    # Format as cards
    cards = []
    for arg in arguments:
        timestamp_formatted = format_timestamp(arg["timestamp"])
        num_speakers = arg.get("num_speakers", 2)
        title = arg.get("title", arg['id'])

        cards.append(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: white;">
            <h3 style="margin: 0 0 8px 0;">{title}</h3>
            <p style="margin: 4px 0; color: #666; font-size: 14px;">
                 {timestamp_formatted}<br>
                 {num_speakers} speakers<br>
                 {arg['id']}
            </p>
            <button onclick="navigator.clipboard.writeText('{arg['id']}')"
                    style="padding: 6px 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
                Copy ID to View
            </button>
        </div>
        """)

    return "".join(cards)

def render_chat_ui(argument_id: str):
    """Render chat-style conversation UI with clickable segments"""
    if not argument_id or not argument_id.strip():
        return "Enter an Argument ID to view the conversation"

    arg = storage.get_argument(argument_id.strip())

    if not arg:
        return f"Argument '{argument_id}' not found"

    # Get conversation segments
    segments = arg.get("conversation_segments", [])

    # Sort segments by start time
    if segments:
        segments = sorted(segments, key=lambda s: s.get('start', 0))

    if not segments:
        # Fallback to old format
        return """
        <div style="padding: 20px; background: #fff3cd; border-radius: 8px;">
            <h3> This argument uses the old format</h3>
            <p>This conversation was recorded before the segment-level analysis feature was added.</p>
            <p>Record a new conversation to see the interactive chat interface with fact-checking!</p>
        </div>
        """

    # Build chat HTML
    timestamp = format_timestamp(arg["timestamp"])
    title = arg.get("title", arg['id'])

    html = f"""
    <style>
        .chat-container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .chat-header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .chat-header h2 {{
            margin: 0 0 8px 0;
        }}
        .chat-header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        .chat-message {{
            display: flex;
            margin-bottom: 20px;
            animation: fadeIn 0.5s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .chat-bubble {{
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chat-bubble:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .speaker-0 .chat-bubble {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-right: auto;
        }}
        .speaker-1 .chat-bubble {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-left: auto;
        }}
        .speaker-label {{
            font-size: 11px;
            opacity: 0.8;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .bubble-text {{
            margin: 0;
            line-height: 1.4;
        }}
        .emotion-badge {{
            display: inline-block;
            font-size: 20px;
            margin-left: 8px;
            vertical-align: middle;
        }}
        .timestamp {{
            font-size: 10px;
            opacity: 0.7;
            margin-top: 4px;
        }}
        .analysis-panel {{
            display: none;
            margin-top: 12px;
            padding: 16px;
            background: white;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            animation: expandPanel 0.3s ease-out;
        }}
        .chat-message:hover .analysis-panel {{
            display: block;
        }}
        @keyframes expandPanel {{
            from {{ opacity: 0; max-height: 0; }}
            to {{ opacity: 1; max-height: 1000px; }}
        }}
        .emotion-detail {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }}
        .emotion-detail h4 {{
            margin: 0 0 8px 0;
            color: #333;
            font-size: 14px;
        }}
        .emotion-detail p {{
            margin: 4px 0;
            font-size: 13px;
            color: #666;
        }}
        .facts-section {{
            margin-top: 12px;
        }}
        .facts-section h4 {{
            margin: 0 0 12px 0;
            color: #333;
            font-size: 14px;
        }}
        .fact-item {{
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #28a745;
            background: #f0f9f4;
            border-radius: 4px;
        }}
        .fact-item.contradicting {{
            border-left-color: #dc3545;
            background: #fcf0f0;
        }}
        .fact-text {{
            margin: 0 0 6px 0;
            font-size: 13px;
            color: #333;
        }}
        .fact-source {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #666;
        }}
        .fact-icon {{
            font-size: 14px;
        }}
        .fact-link {{
            color: #007bff;
            text-decoration: none;
        }}
        .fact-link:hover {{
            text-decoration: underline;
        }}
        .progress-bar {{
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 4px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
    </style>

    <div class="chat-container">
        <div class="chat-header">
            <h2> {title}</h2>
            <p>{timestamp} • {len(segments)} segments</p>
        </div>
    """

    # Render each segment as a chat bubble
    for i, segment in enumerate(segments):
        speaker = segment.get('speaker', 'SPEAKER_00')
        text = segment.get('text', '')
        start = segment.get('start', 0)

        # Get emotion data
        emotion_data = segment.get('emotion', {})
        emotion_label = emotion_data.get('label', 'unknown')
        emotion_conf = emotion_data.get('confidence', 0)
        uncertainty = emotion_data.get('uncertainty', 0)
        speaker_conf = emotion_data.get('speaker_confidence', 0)

        # Get facts
        facts = segment.get('facts', {})
        supporting = facts.get('supporting', [])
        contradicting = facts.get('contradicting', [])

        # Determine speaker class (0 or 1)
        speaker_num = 0 if speaker == 'SPEAKER_00' else 1

        # Build analysis panel HTML
        analysis_html = f"""
        <div class="analysis-panel" id="analysis-{i}">
            <div class="emotion-detail">
                <h4> Emotion Analysis</h4>
                <p><strong>Emotion:</strong> {emotion_label.upper()}</p>
                <p><strong>Confidence:</strong> {emotion_conf:.1%}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {emotion_conf*100}%"></div>
                </div>
                <p><strong>Uncertainty:</strong> {uncertainty:.3f}</p>
                <p><strong>Speaker Confidence:</strong> {speaker_conf:.3f}</p>
            </div>
        """

        # Add fact-checking results - grouped by source type
        if supporting or contradicting:
            # Group all facts by source type
            all_facts = supporting + contradicting
            kb_facts = [f for f in all_facts if f.get('source_type') == 'knowledge_base']
            poly_facts = [f for f in all_facts if f.get('source_type') == 'polymarket']
            web_facts = [f for f in all_facts if f.get('source_type') == 'web']

            # Add Knowledge Base section
            if kb_facts:
                analysis_html += """
                <div class="facts-section">
                    <h4>📚 Knowledge Base</h4>
                """
                for fact in kb_facts:
                    fact_text = fact.get('text', '')
                    source_name = fact.get('source_name', 'Unknown')
                    url = fact.get('url')
                    icon = fact.get('icon', '📚')

                    analysis_html += f"""
                    <div class="fact-item">
                        <p class="fact-text">{fact_text}</p>
                        <div class="fact-source">
                            <span class="fact-icon">{icon}</span>
                            <span>{source_name}</span>
                            {f'<a href="{url}" target="_blank" class="fact-link">View Source</a>' if url else ''}
                        </div>
                    </div>
                    """
                analysis_html += "</div>"

            # Add Polymarket section
            if poly_facts:
                analysis_html += """
                <div class="facts-section">
                    <h4>📊 Polymarket Predictions</h4>
                """
                for fact in poly_facts:
                    fact_text = fact.get('text', '')
                    url = fact.get('url')
                    icon = fact.get('icon', '📊')

                    analysis_html += f"""
                    <div class="fact-item" style="border-left-color: #9c27b0;">
                        <p class="fact-text">{fact_text}</p>
                        <div class="fact-source">
                            <span class="fact-icon">{icon}</span>
                            <span>Prediction Market</span>
                            {f'<a href="{url}" target="_blank" class="fact-link">View Market</a>' if url else ''}
                        </div>
                    </div>
                    """
                analysis_html += "</div>"

            # Add Web Search section
            if web_facts:
                analysis_html += """
                <div class="facts-section">
                    <h4>🌐 Web Sources</h4>
                """
                for fact in web_facts:
                    fact_text = fact.get('text', '')
                    source_name = fact.get('source_name', 'Web')
                    url = fact.get('url')
                    icon = fact.get('icon', '🌐')

                    analysis_html += f"""
                    <div class="fact-item">
                        <p class="fact-text">{fact_text}</p>
                        <div class="fact-source">
                            <span class="fact-icon">{icon}</span>
                            <span>{source_name}</span>
                            {f'<a href="{url}" target="_blank" class="fact-link">View Source</a>' if url else ''}
                        </div>
                    </div>
                    """
                analysis_html += "</div>"
        else:
            # No factual claims detected
            analysis_html += """
            <div class="facts-section">
                <h4>ℹ Fact-Checking</h4>
                <div class="fact-item" style="border-left-color: #6c757d; background: #f8f9fa;">
                    <p class="fact-text">No verifiable factual claims detected in this statement.</p>
                    <p style="margin: 4px 0 0 0; font-size: 11px; color: #999;">
                        This segment appears to be opinion, question, or conversational in nature.
                    </p>
                </div>
            </div>
            """

        analysis_html += "</div>"

        # Build chat message HTML
        html += f"""
        <div class="chat-message speaker-{speaker_num}">
            <div style="width: 100%;">
                <div class="speaker-label">{speaker}</div>
                <div class="chat-bubble" onclick="toggleAnalysis({i})">
                    <p class="bubble-text">
                        {text}
                    </p>
                    <div class="timestamp">
                        {start:.1f}s • {emotion_label.upper()}
                    </div>
                </div>
                {analysis_html}
            </div>
        </div>
        """

    # Add JavaScript for toggle functionality
    html += """
    </div>
    <script>
        function toggleAnalysis(segmentId) {
            const panel = document.getElementById('analysis-' + segmentId);
            if (panel.classList.contains('expanded')) {
                panel.classList.remove('expanded');
            } else {
                // Close all other panels
                document.querySelectorAll('.analysis-panel').forEach(p => {
                    p.classList.remove('expanded');
                });
                // Open clicked panel
                panel.classList.add('expanded');
            }
        }
    </script>
    """

    return html

def search_arguments(query: str):
    """Search arguments by keyword"""
    if not query or not query.strip():
        return "Enter a search query"

    results = storage.search_arguments(query.strip())

    if not results:
        return f"No arguments found containing '{query}'"

    # Format results as cards
    cards = []
    for arg in results:
        timestamp_formatted = format_timestamp(arg["timestamp"])
        title = arg.get("title", arg['id'])

        cards.append(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: white;">
            <h3 style="margin: 0 0 8px 0;">{title}</h3>
            <p style="margin: 4px 0; color: #666; font-size: 14px;">
                 {timestamp_formatted}<br>
                 {arg['id']}
            </p>
            <button onclick="navigator.clipboard.writeText('{arg['id']}')"
                    style="padding: 6px 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
                Copy ID to View
            </button>
        </div>
        """)

    return f"<h3>Found {len(results)} result(s) for \"{query}\"</h3>" + "".join(cards)

def get_stats():
    """Get database statistics"""
    stats = storage.get_stats()
    total = stats.get("total_arguments", 0)

    if total == 0:
        return "No arguments recorded yet"

    winner_dist = stats.get("winner_distribution", {})

    # Format winner distribution
    winner_text = []
    for winner, count in winner_dist.items():
        percentage = (count / total) * 100
        winner_text.append(f"- **{winner}**: {count} ({percentage:.1f}%)")

    return f"""
## Database Statistics

**Total Arguments:** {total}

**Winner Distribution:**
{chr(10).join(winner_text)}

**Latest Recording:** {format_timestamp(stats.get("latest_timestamp", ""))}
"""

# Create Gradio Interface
with gr.Blocks(title="Argument Resolver - Interactive Chat") as app:
    gr.Markdown("""
    #  Argument Resolver - Interactive Chat Browser

    View arguments as interactive chat conversations with emotion analysis and fact-checking.
    Click on any message bubble to see detailed analysis!
    """)

    with gr.Tabs():
        # Tab 1: Browse All
        with gr.Tab(" Browse All"):
            gr.Markdown("### All Recorded Arguments")
            refresh_btn = gr.Button(" Refresh List")
            arguments_display = gr.HTML()

            refresh_btn.click(
                fn=get_arguments_list,
                inputs=[],
                outputs=[arguments_display]
            )

            app.load(
                fn=get_arguments_list,
                inputs=[],
                outputs=[arguments_display]
            )

        # Tab 2: Interactive Chat View
        with gr.Tab(" Chat View"):
            gr.Markdown("### Interactive Conversation View")
            gr.Markdown("Enter an Argument ID to view the conversation as an interactive chat. Click on any message to see emotion analysis and fact-checking!")

            argument_id_input = gr.Textbox(
                label="Argument ID",
                placeholder="e.g., 20251204_132845"
            )
            view_btn = gr.Button("View Conversation", variant="primary")

            chat_display = gr.HTML()

            view_btn.click(
                fn=render_chat_ui,
                inputs=[argument_id_input],
                outputs=[chat_display]
            )

        # Tab 3: Search
        with gr.Tab(" Search"):
            gr.Markdown("### Search Arguments")
            gr.Markdown("Search for arguments containing specific keywords.")

            search_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., climate change, AI, remote work"
            )
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.HTML()

            search_btn.click(
                fn=search_arguments,
                inputs=[search_input],
                outputs=[search_results]
            )

        # Tab 4: Statistics
        with gr.Tab(" Statistics"):
            gr.Markdown("### Database Statistics")
            stats_refresh_btn = gr.Button(" Refresh Stats")
            stats_output = gr.Markdown()

            stats_refresh_btn.click(
                fn=get_stats,
                inputs=[],
                outputs=[stats_output]
            )

            app.load(
                fn=get_stats,
                inputs=[],
                outputs=[stats_output]
            )

    gr.Markdown("""
    ---
    ### Features:

    - **Interactive Chat Bubbles**: Click any message to reveal detailed analysis
    - **Emotion Detection**: See real-time emotion analysis with confidence scores
    - **Multi-Source Fact-Checking**: Facts from knowledge base, Polymarket, and web search
    - **Visual Gradients**: Beautiful color gradients for different speakers
    - **Smooth Animations**: Fade-in effects and hover interactions

    **Recording New Arguments:**
    - Use the Raspberry Pi to record conversations
    - Arguments are automatically processed with emotion detection and fact-checking
    - New conversations appear here instantly
    """)

def main():
    import socket

    def _pick_free_port(prefer=7863):
        try:
            with socket.socket() as s:
                s.bind(("0.0.0.0", prefer))
                return prefer
        except OSError:
            with socket.socket() as s:
                s.bind(("0.0.0.0", 0))
                return s.getsockname()[1]

    port = _pick_free_port(7863)

    print(f"\n{'='*60}")
    print(f" ARGUMENT RESOLVER - INTERACTIVE CHAT UI")
    print(f"{'='*60}")
    print(f"Open in browser: http://0.0.0.0:{port}")
    print(f"Database: {storage.base_dir}")
    print(f"Total arguments: {storage.get_stats().get('total_arguments', 0)}")
    print(f"{'='*60}\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=False
    )

if __name__ == "__main__":
    main()
