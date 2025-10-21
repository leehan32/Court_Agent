from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# 콘솔 객체 생성
console = Console()

def print_header(title: str):
    """프로그램 시작 헤더를 출력합니다."""
    console.print(Rule(f"[bold cyan]⚖️ {title} ⚖️[/bold cyan]"))

def print_judge_panel(judges: list):
    """이번 재판의 서브 판사 구성을 패널로 출력합니다."""
    judge_text = "\n".join([f"  - {i+1}: {judge['name']}" for i, judge in enumerate(judges)])
    console.print(
        Panel(
            Text(judge_text, justify="left"),
            title="[bold yellow]이번 재판의 서브 판사 구성[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        )
    )

def print_turn_header(turn_count: int):
    """토론 턴 헤더를 출력합니다."""
    console.print(Rule(f"[bold]변호사 토론 (턴 {turn_count})[/bold]"))

def print_speech(speaker: str, speech: str):
    """에이전트의 발언을 패널로 출력합니다."""
    title = f"[bold magenta]{speaker}[/bold magenta]"
    if "판사" in speaker:
        title = f"[bold green]{speaker}[/bold green]"
    
    console.print(
        Panel(
            speech,
            title=title,
            border_style="white",
            padding=(1, 2)
        )
    )

def print_verdict_header(title: str):
    """판결 헤더를 출력합니다."""
    console.print(Rule(f"[bold red]{title}[/bold red]"))

def print_final_verdict(verdict: str):
    """최종 판결문을 패널로 출력합니다."""
    console.print(
        Panel(
            Text(verdict, justify="center"),
            title="[bold red]최종 판결[/bold red]",
            border_style="red"
        )
    )

def print_update_header():
    """DB 업데이트 헤더를 출력합니다."""
    console.print(Rule("[bold blue]변호사 에이전트 지식 베이스 업데이트[/bold blue]"))

def print_lesson(lawyer: str, outcome: str, lesson: str):
    """학습된 교훈을 출력합니다."""
    emoji = "✅" if outcome == "승리" else ("❌" if outcome == "패배" else "🟡")
    console.print(f"{emoji} [bold]{lawyer} ({outcome})[/bold] -> 학습된 교훈: {lesson}")